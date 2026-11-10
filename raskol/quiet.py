import asyncio
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from random import choice
from time import monotonic, time

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter

from .conversation import SPONTANEOUS_COOLDOWN, Conversation, Line

LOGGER = logging.getLogger("raskol")
QUIET_NUDGES = (
    "Чё, блять, молчим? Кто там ещё живой?",
    "Какого хуя тут так тихо? Чё вы там делаете?",
    "Вы чё, блять, все в режим кирпича ушли? Как день-то?",
    "Ебать, какая тишина. Вы там работаете или чат закопали?",
    "Ну и чё притихли, блять? Хоть один скажите, как дела.",
    "Чё за хуйня, тут перекличку проводить уже? Кто чем занят?",
)


@contextmanager
def open_quiet_state(
    state_path: Path, bot_id: int, chat_ids: set[int]
) -> Iterator[sqlite3.Connection]:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(state_path)) as db:
        with db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS quiet_chats (
                    bot_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    last_activity INTEGER NOT NULL,
                    nudged INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (bot_id, chat_id)
                )
                """
            )
            started_at = int(time())
            db.executemany(
                "INSERT OR IGNORE INTO quiet_chats (bot_id, chat_id, last_activity) VALUES (?, ?, ?)",
                ((bot_id, chat_id, started_at) for chat_id in chat_ids),
            )
        yield db


def note_human_activity(db: sqlite3.Connection, bot_id: int, chat_id: int, timestamp: int) -> None:
    with db:
        db.execute(
            """
            UPDATE quiet_chats SET last_activity = ?, nudged = 0
            WHERE bot_id = ? AND chat_id = ? AND last_activity < ?
            """,
            (timestamp, bot_id, chat_id, timestamp),
        )


def claim_quiet_nudge(
    db: sqlite3.Connection, bot_id: int, chat_id: int, cutoff: int
) -> int | None:
    activity = db.execute(
        """
        SELECT last_activity FROM quiet_chats
        WHERE bot_id = ? AND chat_id = ? AND last_activity <= ? AND nudged = 0
        """,
        (bot_id, chat_id, cutoff),
    ).fetchone()
    if activity is None:
        return None

    timestamp = activity[0]
    # Reserve this stretch of silence before sending so a restart cannot duplicate it.
    with db:
        result = db.execute(
            """
            UPDATE quiet_chats SET nudged = 1
            WHERE bot_id = ? AND chat_id = ? AND last_activity = ? AND nudged = 0
            """,
            (bot_id, chat_id, timestamp),
        )
    return timestamp if result.rowcount == 1 else None


async def run_quiet_nudges(
    bot: Bot,
    bot_name: str,
    chat_ids: set[int],
    rooms: dict[tuple[int, int | None], Conversation],
    db: sqlite3.Connection,
    idle_after: int,
) -> None:
    while True:
        await asyncio.sleep(15)
        for chat_id in chat_ids:
            room = rooms.get((chat_id, None))
            if room is not None and monotonic() - room.last_sent_at < SPONTANEOUS_COOLDOWN:
                continue
            timestamp = claim_quiet_nudge(db, bot.id, chat_id, int(time()) - idle_after)
            if timestamp is None:
                continue

            text = choice(QUIET_NUDGES)
            try:
                sent = await bot.send_message(chat_id=chat_id, text=text)
            except TelegramRetryAfter as exc:
                with db:
                    db.execute(
                        """
                        UPDATE quiet_chats SET nudged = 0
                        WHERE bot_id = ? AND chat_id = ? AND last_activity = ?
                        """,
                        (bot.id, chat_id, timestamp),
                    )
                await asyncio.sleep(exc.retry_after)
            except TelegramAPIError as exc:
                LOGGER.warning("Окрик после тишины не подтверждён (%s)", type(exc).__name__)
            else:
                room = rooms.setdefault((chat_id, None), Conversation())
                room.sent(Line(sent.message_id, bot_name, text, bot=True), monotonic())
