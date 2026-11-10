import asyncio
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, time, timedelta
from pathlib import Path
from random import choice
from time import monotonic
from zoneinfo import ZoneInfo

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter

from .conversation import Conversation, Line

LOGGER = logging.getLogger("raskol")
MORNING_WINDOW = timedelta(hours=2)
GREETINGS = (
    "чайники",
    "сонные табуретки",
    "будильники ебучие",
    "пельмени в тапках",
    "ходячие термосы",
    "крокодилы в халатах",
    "люди с лицами на зарядке",
    "клуб помятых подушек",
    "ценители вчерашнего чая",
    "сонная хуйня на ножках",
)


def greeting_day(now: datetime, at: time) -> str | None:
    scheduled = datetime.combine(now.date(), at, now.tzinfo)
    if timedelta(0) <= now - scheduled < MORNING_WINDOW:
        return now.date().isoformat()
    return None


def claim_greeting(db: sqlite3.Connection, bot_id: int, chat_id: int, day: str) -> bool:
    # Persist the reservation before calling Telegram, whose send API cannot deduplicate retries.
    with db:
        result = db.execute(
            """
            INSERT INTO morning_greetings (bot_id, chat_id, day) VALUES (?, ?, ?)
            ON CONFLICT (bot_id, chat_id) DO UPDATE SET day = excluded.day
            WHERE morning_greetings.day < excluded.day
            """,
            (bot_id, chat_id, day),
        )
    return result.rowcount == 1


async def run_mornings(
    bot: Bot,
    bot_name: str,
    chat_ids: set[int],
    rooms: dict[tuple[int, int | None], Conversation],
    at: time,
    timezone: ZoneInfo,
    state_path: Path,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(state_path)) as db:
        with db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS morning_greetings (
                    bot_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    PRIMARY KEY (bot_id, chat_id)
                )
                """
            )

        while True:
            for chat_id in sorted(chat_ids):
                day = greeting_day(datetime.now(timezone), at)
                if day is None:
                    break
                if not claim_greeting(db, bot.id, chat_id, day):
                    continue

                text = f"Доброе утро, {choice(GREETINGS)}"
                try:
                    sent = await bot.send_message(chat_id=chat_id, text=text)
                except TelegramRetryAfter as exc:
                    with db:
                        db.execute(
                            "DELETE FROM morning_greetings WHERE bot_id=? AND chat_id=? AND day=?",
                            (bot.id, chat_id, day),
                        )
                    await asyncio.sleep(exc.retry_after)
                except TelegramAPIError as exc:
                    LOGGER.warning("Утреннее сообщение не подтверждено (%s)", type(exc).__name__)
                else:
                    room = rooms.setdefault((chat_id, None), Conversation())
                    room.sent(Line(sent.message_id, bot_name, text, bot=True), monotonic())

            await asyncio.sleep(15)
