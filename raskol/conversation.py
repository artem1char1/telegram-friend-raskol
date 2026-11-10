from collections import deque
from dataclasses import dataclass, field
from typing import Literal

HISTORY_SIZE = 20
MIN_MESSAGES = 3
DIRECT_DELAY = 3
AMBIENT_PAUSE = 8
AMBIENT_WINDOW = 90
AMBIENT_CHECK_INTERVAL = 60
SPONTANEOUS_COOLDOWN = 300
IDLE_INTERVAL = 24 * 60 * 60

Mode = Literal["direct", "ambient", "idle"]


@dataclass(frozen=True)
class Line:
    message_id: int
    author: str
    text: str
    bot: bool = False


@dataclass(frozen=True)
class Turn:
    mode: Mode
    reply_to: int | None
    latest_human_id: int
    history: tuple[Line, ...]


@dataclass
class Conversation:
    history: deque[Line] = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    latest_human_id: int | None = None
    last_human_at: float = float("-inf")
    last_sent_at: float = float("-inf")
    last_ambient_attempt: float = float("-inf")
    last_idle_attempt: float = float("-inf")
    messages_since_turn: int = 0
    direct_id: int | None = None
    direct_at: float = 0

    def receive(self, line: Line, now: float, *, direct: bool) -> None:
        self.history.append(line)
        self.latest_human_id = line.message_id
        self.last_human_at = now
        self.messages_since_turn += 1
        if direct:
            self.direct_id = line.message_id
            self.direct_at = now

    def take_turn(self, now: float, idle_after: int) -> Turn | None:
        if self.latest_human_id is None:
            return None

        if self.direct_id is not None:
            if now - self.direct_at < DIRECT_DELAY:
                return None
            reply_to = self.direct_id
            self.direct_id = None
            self.messages_since_turn = 0
            return Turn("direct", reply_to, self.latest_human_id, tuple(self.history))

        if now - self.last_sent_at < SPONTANEOUS_COOLDOWN:
            return None

        quiet_for = now - self.last_human_at
        if (
            idle_after > 0
            and len(self.history) >= MIN_MESSAGES
            and quiet_for >= idle_after
            and now - self.last_sent_at >= idle_after
            and now - self.last_idle_attempt >= IDLE_INTERVAL
            and self.last_human_at > self.last_idle_attempt
        ):
            self.last_idle_attempt = now
            self.messages_since_turn = 0
            return Turn("idle", None, self.latest_human_id, tuple(self.history))

        if (
            self.messages_since_turn >= MIN_MESSAGES
            and AMBIENT_PAUSE <= quiet_for <= AMBIENT_WINDOW
            and now - self.last_ambient_attempt >= AMBIENT_CHECK_INTERVAL
        ):
            self.last_ambient_attempt = now
            self.messages_since_turn = 0
            return Turn("ambient", self.latest_human_id, self.latest_human_id, tuple(self.history))

        return None

    def is_current(self, turn: Turn) -> bool:
        if turn.mode == "direct":
            return self.direct_id is None
        return self.latest_human_id == turn.latest_human_id

    def sent(self, line: Line, now: float) -> None:
        self.history.append(line)
        self.last_sent_at = now
