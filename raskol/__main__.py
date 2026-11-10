import asyncio
import logging
import os
from datetime import UTC, datetime
from time import monotonic

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType, MessageEntityType
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message, ReplyParameters
from aiohttp import ClientSession, ClientTimeout
from dotenv import load_dotenv

from .conversation import Conversation, Line, Turn
from .llm import DEFAULT_MODELS, LanguageModel, ModelError

LOGGER = logging.getLogger("raskol")
RoomKey = tuple[int, int | None]


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
    return False


async def deliver(
    bot: Bot,
    bot_name: str,
    model: LanguageModel,
    key: RoomKey,
    room: Conversation,
    turn: Turn,
) -> None:
    try:
        text = await model.respond(turn)
        if not text or not room.is_current(turn):
            return

        chat_id, topic_id = key
        reply = ReplyParameters(message_id=turn.reply_to) if turn.reply_to is not None else None
        sent = await bot.send_message(
            chat_id=chat_id,
            message_thread_id=topic_id,
            reply_parameters=reply,
            text=text,
        )
        room.sent(Line(sent.message_id, bot_name, text, bot=True), monotonic())
    except ModelError as exc:
        LOGGER.warning("Модель не ответила в режиме %s (%s)", turn.mode, exc)
    except TelegramAPIError as exc:
        LOGGER.warning("Telegram не принял сообщение (%s)", type(exc).__name__)


async def run_turns(
    bot: Bot,
    bot_name: str,
    model: LanguageModel,
    rooms: dict[RoomKey, Conversation],
    idle_after: int,
) -> None:
    active: set[RoomKey] = set()

    async def send_turn(key: RoomKey, room: Conversation, turn: Turn) -> None:
        try:
            await deliver(bot, bot_name, model, key, room, turn)
        finally:
            active.remove(key)

    async with asyncio.TaskGroup() as tasks:
        while True:
            now = monotonic()
            for key, room in rooms.items():
                if key in active:
                    continue
                turn = room.take_turn(now, idle_after)
                if turn is not None:
                    active.add(key)
                    tasks.create_task(send_turn(key, room, turn))
            await asyncio.sleep(1)


def create_router(
    rooms: dict[RoomKey, Conversation], chat_ids: set[int], bot_id: int, username: str
) -> Router:
    router = Router()
    router.message.filter(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))

    @router.message(Command("chatid"))
    async def show_chat_id(message: Message) -> None:
        await message.answer(f"ID группы: {message.chat.id}")

    @router.message(F.text)
    async def receive_message(message: Message) -> None:
        if message.chat.id not in chat_ids or not message.from_user or message.from_user.is_bot:
            return
        if not message.text or message.text.startswith("/"):
            return
        if (datetime.now(UTC) - message.date).total_seconds() > 120:
            return

        topic_id = message.message_thread_id if message.is_topic_message else None
        key = (message.chat.id, topic_id)
        if key not in rooms:
            rooms[key] = Conversation()
        rooms[key].receive(
            Line(message.message_id, message.from_user.first_name[:64], message.text[:400]),
            monotonic(),
            direct=addresses_bot(message, bot_id, username),
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
        idle_minutes = int(os.getenv("RASKOL_IDLE_MINUTES", "180"))
        if any(chat_id >= 0 for chat_id in chat_ids) or idle_minutes < 0:
            raise ValueError
    except ValueError:
        raise SystemExit(
            "Укажи отрицательные ID групп через запятую и неотрицательное число минут тишины."
        ) from None

    if provider == "groq" and chat_ids and not groq_key:
        raise SystemExit("Укажи GROQ_API_KEY в .env или выбери LLM_PROVIDER=ollama.")
    if not chat_ids:
        LOGGER.info("Режим настройки. Отправь /chatid в группе, чтобы узнать её ID.")

    rooms: dict[RoomKey, Conversation] = {}
    timeout = ClientTimeout(total=30 if provider == "groq" else 120)
    async with Bot(token) as bot, ClientSession(timeout=timeout) as session:
        me = await bot.get_me()
        model = LanguageModel(session, provider, model_name, groq_key, ollama_url)
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router(rooms, chat_ids, me.id, me.username or ""))
        async with asyncio.TaskGroup() as tasks:
            scheduler = tasks.create_task(run_turns(bot, me.first_name, model, rooms, idle_minutes * 60))
            try:
                await dispatcher.start_polling(
                    bot, allowed_updates=["message"], handle_as_tasks=False, close_bot_session=False
                )
            finally:
                scheduler.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
