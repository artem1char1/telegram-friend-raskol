import asyncio
import logging
import os
import re
import sqlite3
from contextlib import AsyncExitStack
from datetime import UTC, datetime, time
from pathlib import Path
from random import choice
from time import monotonic
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType, MessageEntityType
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message, ReplyParameters
from aiohttp import ClientSession, ClientTimeout
from dotenv import load_dotenv

from .conversation import Conversation, Line, Reminder, Turn
from .llm import DEFAULT_MODELS, LanguageModel, ModelError
from .morning import run_mornings
from .quiet import note_human_activity, open_quiet_state, run_quiet_nudges

LOGGER = logging.getLogger("raskol")
RoomKey = tuple[int, int | None]
IGNORED_LINES = (
    "Алё, блять, я в стену это сказал?",
    "Ты там живой? Или мне, блять, самому себе отвечать?",
    "Чё, блять, моя реплика в пустоту улетела? Ау.",
    "Алё, я тут вообще-то с тобой говорю. Чё за тишина, бля?",
    "Ты пропал, блять? Я тут ответ жду вообще-то.",
)
BOT_NAME = r"раскол(?:ьников)?"
BOT_NAME_WORD = re.compile(rf"\b{BOT_NAME}\b", re.IGNORECASE)
BOT_NAME_ADDRESS = re.compile(
    rf"^\s*{BOT_NAME}(?="
    r"\s*(?:[,.!?:…—]|$)|"
    r"\s+(?:ты|тебе|тебя|тобой|твой|твоя|твоё|твое|твои|"
    r"ч[еёо]|что|как|где|куда|когда|почему|зачем|привет|здарова|здорово|скажи|расскажи)"
    r"(?=\s|[,.!?:…]|$))"
    rf"|[,—]\s*{BOT_NAME}(?=\s*(?:[,.!?:…—]|$))"
    rf"|\b(?:думы|думаешь|скажешь|считаешь|делаешь|молчишь|живой)\s+{BOT_NAME}(?=\s*(?:[!?…]|$))"
    rf"|^\s*(?:эй|йо|ал[её]|слушай|привет|здарова|здорово)[\s,!:…—]+{BOT_NAME}\b",
    re.IGNORECASE,
)
POSSESSIVE = r"тво(?:й|я|[её]|и|его|ей|ю|им|их|ими|ему|[её]м)"
SECOND_PERSON = re.compile(rf"\b(?:ты|тебя|тебе|тобо[йю]|{POSSESSIVE})\b", re.IGNORECASE)
ARTIST_ADDRESS = re.compile(
    rf"\b{POSSESSIVE}\s+(?:трек|альбом|фит|музык|песн|творчеств)\w*\b",
    re.IGNORECASE,
)
STATE_PATH = Path(__file__).resolve().parent.parent / ".raskol" / "state.sqlite3"


def addresses_bot(message: Message, bot_id: int, username: str) -> bool:
    replied = message.reply_to_message
    if replied and replied.from_user and replied.from_user.id == bot_id:
        return True

    for entity in message.entities or []:
        if entity.type == MessageEntityType.TEXT_MENTION and entity.user and entity.user.id == bot_id:
            return True
        if entity.type == MessageEntityType.MENTION and message.text:
            if entity.extract_from(message.text).casefold() == f"@{username}".casefold():
                return True
    return bool(message.text and BOT_NAME_ADDRESS.search(message.text))


async def deliver(
    bot: Bot,
    bot_name: str,
    model: LanguageModel,
    key: RoomKey,
    room: Conversation,
    turn: Turn,
    ignored_after: int = 0,
) -> None:
    try:
        answer = await model.respond(turn)
        if not room.is_current(turn, monotonic()):
            return
        if not answer.text:
            room.finish_turn(turn, monotonic(), sent=False, keep_talking=answer.keep_talking)
            return

        chat_id, topic_id = key
        sent = await bot.send_message(
            chat_id=chat_id,
            message_thread_id=topic_id,
            reply_parameters=ReplyParameters(message_id=turn.reply_to),
            text=answer.text,
        )
        now = monotonic()
        room.sent(Line(sent.message_id, bot_name, answer.text, bot=True), now)
        room.finish_turn(turn, now, sent=True, keep_talking=answer.keep_talking)
        room.reminder = None
        if (
            ignored_after > 0
            and answer.keep_talking
            and turn.mode != "ambient"
            and turn.participant_id is not None
        ):
            room.reminder = Reminder(sent.message_id, turn.participant_id, now + ignored_after)
        LOGGER.info("Ответ отправлен (%s)", turn.mode)
    except ModelError as exc:
        LOGGER.warning("Модель не ответила в режиме %s (%s)", turn.mode, exc)
    except TelegramAPIError as exc:
        LOGGER.warning("Telegram не принял сообщение (%s)", type(exc).__name__)


async def send_reminder(
    bot: Bot, bot_name: str, key: RoomKey, room: Conversation, reminder: Reminder
) -> None:
    if room.reminder is not reminder or room.pending is not None:
        return
    room.reminder = None

    chat_id, topic_id = key
    text = choice(IGNORED_LINES)
    try:
        sent = await bot.send_message(
            chat_id=chat_id,
            message_thread_id=topic_id,
            reply_parameters=ReplyParameters(message_id=reminder.message_id),
            text=text,
        )
    except TelegramAPIError as exc:
        LOGGER.warning("Telegram не принял повторный окрик (%s)", type(exc).__name__)
        return

    now = monotonic()
    room.sent(Line(sent.message_id, bot_name, text, bot=True), now)
    room.remember_partner(reminder.participant_id, now, keep_talking=True)
    LOGGER.info("Окрик после отсутствия ответа отправлен")


async def run_turns(
    bot: Bot,
    bot_name: str,
    model: LanguageModel,
    rooms: dict[RoomKey, Conversation],
    ambient_chance: float,
    ignored_after: int,
) -> None:
    active: set[RoomKey] = set()

    async def send_turn(key: RoomKey, room: Conversation, turn: Turn | Reminder) -> None:
        try:
            if isinstance(turn, Reminder):
                await send_reminder(bot, bot_name, key, room, turn)
            else:
                await deliver(bot, bot_name, model, key, room, turn, ignored_after)
        finally:
            active.remove(key)

    async with asyncio.TaskGroup() as tasks:
        while True:
            now = monotonic()
            for key, room in rooms.items():
                if key in active:
                    continue
                turn: Turn | Reminder | None = room.take_turn(now, ambient_chance)
                reminder = room.reminder
                if turn is None and room.pending is None and reminder and now >= reminder.due_at:
                    turn = reminder
                if turn is not None:
                    active.add(key)
                    tasks.create_task(send_turn(key, room, turn))
            await asyncio.sleep(1)


def create_router(
    rooms: dict[RoomKey, Conversation],
    chat_ids: set[int],
    bot_id: int,
    username: str,
    quiet_state: sqlite3.Connection | None = None,
) -> Router:
    router = Router()
    router.message.filter(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))

    @router.message(Command("chatid"))
    async def show_chat_id(message: Message) -> None:
        await message.answer(f"ID группы: {message.chat.id}")

    @router.message()
    async def receive_message(message: Message) -> None:
        if message.chat.id not in chat_ids or not message.from_user or message.from_user.is_bot:
            return
        if quiet_state is not None:
            note_human_activity(quiet_state, bot_id, message.chat.id, int(message.date.timestamp()))
        if (datetime.now(UTC) - message.date).total_seconds() > 120:
            return

        topic_id = message.message_thread_id if message.is_topic_message else None
        key = (message.chat.id, topic_id)
        room = rooms.get(key)
        if room is not None:
            replied = message.reply_to_message
            room.note_response(
                message.from_user.id,
                replied.message_id if replied else None,
                reply_to_bot=bool(replied and replied.from_user and replied.from_user.id == bot_id),
            )
        if not message.text or message.text.startswith("/"):
            return

        if room is None:
            room = rooms[key] = Conversation()
        second_person = bool(SECOND_PERSON.search(message.text))
        room.receive(
            Line(
                message.message_id,
                message.from_user.first_name[:64],
                message.text[:400],
                user_id=message.from_user.id,
            ),
            monotonic(),
            direct=addresses_bot(message, bot_id, username),
            can_followup=message.reply_to_message is None,
            second_person=second_person,
            named=bool(BOT_NAME_WORD.search(message.text)),
            possible_address=bool(ARTIST_ADDRESS.search(message.text)),
        )

    return router


async def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Укажи TELEGRAM_BOT_TOKEN в .env.")

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider != "groq" and provider != "ollama":
        raise SystemExit("Укажи groq или ollama в LLM_PROVIDER.")
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").strip().rstrip("/")

    try:
        chat_ids = {
            int(value) for value in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if value.strip()
        }
        idle_minutes = int(os.getenv("RASKOL_IDLE_MINUTES", "720"))
        if any(chat_id >= 0 for chat_id in chat_ids) or idle_minutes < 0:
            raise ValueError
    except ValueError:
        raise SystemExit(
            "Укажи отрицательные ID групп через запятую и неотрицательное число минут тишины."
        ) from None

    try:
        ambient_chance = float(os.getenv("RASKOL_AMBIENT_CHANCE", "0.08"))
        if not 0 <= ambient_chance <= 1:
            raise ValueError
    except ValueError:
        raise SystemExit("Укажи RASKOL_AMBIENT_CHANCE числом от 0 до 1.") from None

    try:
        ignored_minutes = int(os.getenv("RASKOL_IGNORED_MINUTES", "10"))
        if ignored_minutes < 0:
            raise ValueError
    except ValueError:
        raise SystemExit("Укажи RASKOL_IGNORED_MINUTES целым числом от 0.") from None

    morning_setting = os.getenv("RASKOL_MORNING_TIME", "09:00").strip().lower()
    morning_at = None
    timezone = None
    if morning_setting != "off":
        try:
            morning_at = time.fromisoformat(morning_setting)
            if not re.fullmatch(r"\d{2}:\d{2}", morning_setting):
                raise ValueError
            timezone = ZoneInfo(os.getenv("RASKOL_TIMEZONE", "Europe/Moscow").strip())
        except (ValueError, ZoneInfoNotFoundError):
            raise SystemExit(
                "Укажи RASKOL_MORNING_TIME в формате ЧЧ:ММ или off и часовой пояс "
                "RASKOL_TIMEZONE, например Europe/Moscow."
            ) from None

    if provider == "groq" and chat_ids and not groq_key:
        raise SystemExit("Укажи GROQ_API_KEY в .env или выбери LLM_PROVIDER=ollama.")
    if not chat_ids:
        LOGGER.info("Режим настройки. Отправь /chatid в группе, чтобы узнать её ID.")

    rooms: dict[RoomKey, Conversation] = {}
    timeout = ClientTimeout(total=30 if provider == "groq" else 120)
    async with Bot(token) as bot, ClientSession(timeout=timeout) as session, AsyncExitStack() as stack:
        me = await bot.get_me()
        model = LanguageModel(session, provider, model_name, groq_key, ollama_url)
        quiet_state = (
            stack.enter_context(open_quiet_state(STATE_PATH, me.id, chat_ids))
            if chat_ids and idle_minutes
            else None
        )
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router(rooms, chat_ids, me.id, me.username or "", quiet_state))
        async with asyncio.TaskGroup() as tasks:
            schedulers = [
                tasks.create_task(
                    run_turns(bot, me.first_name, model, rooms, ambient_chance, ignored_minutes * 60)
                )
            ]
            if quiet_state is not None:
                schedulers.append(
                    tasks.create_task(
                        run_quiet_nudges(bot, me.first_name, chat_ids, rooms, quiet_state, idle_minutes * 60)
                    )
                )
            if chat_ids and morning_at is not None and timezone is not None:
                schedulers.append(
                    tasks.create_task(
                        run_mornings(
                            bot, me.first_name, chat_ids, rooms, morning_at, timezone, STATE_PATH
                        )
                    )
                )
            try:
                await dispatcher.start_polling(
                    bot, allowed_updates=["message"], handle_as_tasks=False, close_bot_session=False
                )
            finally:
                for scheduler in schedulers:
                    scheduler.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
