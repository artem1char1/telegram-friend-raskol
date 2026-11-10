from collections import deque
from dataclasses import dataclass, field
from random import random
from typing import Literal

HISTORY_SIZE = 20
MIN_MESSAGES = 3
DIRECT_DELAY = 3
DIALOGUE_TIMEOUT = 3 * 60
AMBIENT_PAUSE = 8
AMBIENT_WINDOW = 90
AMBIENT_CHECK_INTERVAL = 60
SPONTANEOUS_COOLDOWN = 20 * 60

Mode = Literal["direct", "followup", "ambient"]


@dataclass(frozen=True)
class Line:
    message_id: int
    author: str
    text: str
    bot: bool = False
    user_id: int | None = None


@dataclass(frozen=True)
class Turn:
    mode: Mode
    reply_to: int
    latest_human_id: int
    history: tuple[Line, ...]
    participant_id: int | None = None


@dataclass(frozen=True)
class PendingReply:
    mode: Literal["direct", "followup"]
    line: Line
    received_at: float


@dataclass
class Conversation:
    history: deque[Line] = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    latest_human: Line | None = None
    last_human_at: float = float("-inf")
    last_sent_at: float = float("-inf")
    last_ambient_attempt: float = float("-inf")
    messages_since_turn: int = 0
    pending: PendingReply | None = None
    dialogue_user_id: int | None = None
    dialogue_until: float = 0

    def in_dialogue(self, now: float) -> bool:
        return self.dialogue_user_id is not None and now < self.dialogue_until

    def receive(self, line: Line, now: float, *, direct: bool, can_followup: bool) -> None:
        self.history.append(line)
        self.latest_human = line
        self.last_human_at = now
        self.messages_since_turn += 1
        if direct:
            self.pending = PendingReply("direct", line, now)
        elif (
            can_followup
            and self.in_dialogue(now)
            and line.user_id == self.dialogue_user_id
            and (self.pending is None or self.pending.mode == "followup")
        ):
            self.pending = PendingReply("followup", line, now)

    def take_turn(self, now: float, ambient_chance: float) -> Turn | None:
        latest = self.latest_human
        if latest is None:
            return None

        pending = self.pending
        if pending is not None:
            if now - pending.received_at < DIRECT_DELAY:
                return None
            self.pending = None
            if pending.mode == "direct" or (
                self.in_dialogue(pending.received_at)
                and pending.line.user_id == self.dialogue_user_id
            ):
                self.messages_since_turn = 0
                history = tuple(self.history)
                if pending.line not in history:
                    history = (pending.line, *history[1:])
                return Turn(
                    pending.mode, pending.line.message_id, latest.message_id, history,
                    pending.line.user_id,
                )

        if self.in_dialogue(now) or now - self.last_sent_at < SPONTANEOUS_COOLDOWN:
            return None

        quiet_for = now - self.last_human_at
        if (
            ambient_chance > 0
            and self.messages_since_turn >= MIN_MESSAGES
            and AMBIENT_PAUSE <= quiet_for <= AMBIENT_WINDOW
            and now - self.last_ambient_attempt >= AMBIENT_CHECK_INTERVAL
        ):
            self.last_ambient_attempt = now
            self.messages_since_turn = 0
            if random() < ambient_chance:
                return Turn(
                    "ambient", latest.message_id, latest.message_id, tuple(self.history), latest.user_id
                )

        return None

    def is_current(self, turn: Turn, now: float) -> bool:
        if turn.mode == "direct":
            return self.pending is None or self.pending.mode != "direct"
        if turn.mode == "followup":
            return self.pending is None
        return (
            self.latest_human is not None
            and self.latest_human.message_id == turn.latest_human_id
            and now - self.last_sent_at >= SPONTANEOUS_COOLDOWN
        )

    def sent(self, line: Line, now: float) -> None:
        self.history.append(line)
        self.last_sent_at = now

    def finish_turn(self, turn: Turn, now: float, *, sent: bool, keep_talking: bool) -> None:
        if turn.participant_id is None:
            return
        if sent and keep_talking:
            self.dialogue_user_id = turn.participant_id
            self.dialogue_until = now + DIALOGUE_TIMEOUT
        elif not keep_talking and self.dialogue_user_id == turn.participant_id:
            self.dialogue_user_id = None
            if self.pending is not None and self.pending.mode == "followup":
                self.pending = None
