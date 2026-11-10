from datetime import UTC, datetime
from time import monotonic
from unittest import IsolatedAsyncioTestCase

from aiogram import Bot, Dispatcher
from aiogram.types import Update

from raskol.__main__ import create_router


class AddressingTest(IsolatedAsyncioTestCase):
    async def route(self, text: str):
        rooms = {}
        async with Bot("123456:synthetic") as bot:
            dispatcher = Dispatcher()
            dispatcher.include_router(create_router(rooms, {-100}, bot.id, "RaskolBot"))
            update = Update.model_validate({
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": datetime.now(UTC),
                    "chat": {"id": -100, "type": "group"},
                    "from": {"id": 2, "first_name": "Тест", "is_bot": False},
                    "text": text,
                },
            })
            await dispatcher.feed_update(bot, update)
        return rooms[(-100, None)].take_turn(monotonic() + 4, ambient_chance=0)

    async def test_direct_addresses(self):
        for text in ("эй раскол бля", "раскол ты тут ?", "Чё думы, раскол?"):
            with self.subTest(text=text):
                turn = await self.route(text)
                self.assertIsNotNone(turn)
                self.assertEqual(turn.mode, "direct")
                self.assertEqual(turn.reply_to, 1)

    async def test_other_name_positions_get_contextual_check(self):
        for text in ("ну раскол бля", "просыпайся раскольников", "Раскол выпустил трек"):
            with self.subTest(text=text):
                turn = await self.route(text)
                self.assertIsNotNone(turn)
                self.assertEqual(turn.mode, "named")

    async def test_unrelated_chat_does_not_schedule_a_reply(self):
        self.assertIsNone(await self.route("Ребят, кто сегодня на работе?"))
        self.assertIsNone(await self.route("Мне надо расколоть дрова"))
