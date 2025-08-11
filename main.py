import os
import aiohttp
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery
from aiogram.filters import CommandStart
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CRYPTOBOT_TOKEN = os.getenv("CRYPTOBOT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ====== Генерация клавиатуры для покупки токенов ======
def get_buy_tokens_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="100 токенов - $1", callback_data="buy_100")],
            [InlineKeyboardButton(text="500 токенов - $5", callback_data="buy_500")],
            [InlineKeyboardButton(text="1000 токенов - $10", callback_data="buy_1000")],
        ]
    )

# ====== Старт ======
@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "Привет! Выберите пакет токенов для покупки:",
        reply_markup=get_buy_tokens_keyboard()
    )

# ====== Создание платежа в USD ======
async def create_invoice(amount: int, user_id: int):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://pay.crypt.bot/api/createInvoice",
            headers={"Crypto-Pay-API-Token": CRYPTOBOT_TOKEN},
            json={
                "asset": "USDT",
                "amount": amount,
                "description": f"Покупка {amount} USD токенов",
                "hidden_message": f"Спасибо за покупку токенов, пользователь {user_id}",
                "expires_in": 3600
            }
        ) as resp:
            data = await resp.json()
            if data.get("ok"):
                return data["result"]["pay_url"]
            else:
                print("Ошибка создания счета:", data)
                return None

# ====== Обработка нажатия кнопки покупки ======
@dp.callback_query(F.data.startswith("buy_"))
async def process_buy(callback: CallbackQuery):
    amount_usd = int(callback.data.split("_")[1]) // 100  # 100 токенов = $1
    pay_url = await create_invoice(amount_usd, callback.from_user.id)

    if pay_url:
        await callback.message.answer(
            f"Оплатите по ссылке:\n{pay_url}\nПосле оплаты нажмите кнопку 'Проверить оплату'.",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="Проверить оплату", callback_data=f"check_{amount_usd}")]
                ]
            )
        )
    else:
        await callback.message.answer("Ошибка при создании платежа.")

# ====== Проверка оплаты ======
async def check_payment_status(invoice_id: int):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://pay.crypt.bot/api/getInvoices?invoice_ids={invoice_id}",
            headers={"Crypto-Pay-API-Token": CRYPTOBOT_TOKEN}
        ) as resp:
            data = await resp.json()
            if data.get("ok") and data["result"]["items"]:
                return data["result"]["items"][0]["status"]
            return None

@dp.callback_query(F.data.startswith("check_"))
async def process_check(callback: CallbackQuery):
    # Тут нужно подставить реальный invoice_id, если ты его сохраняешь
    invoice_id = 123456  # пример
    status = await check_payment_status(invoice_id)
    if status == "paid":
        await callback.message.answer("✅ Оплата получена! Токены начислены.")
    else:
        await callback.message.answer("⏳ Оплата не найдена. Попробуйте позже.")

# ====== Запуск ======
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
