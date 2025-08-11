import logging
import asyncio
import openai
import aiohttp
import os
import json
import re
from typing import Dict, Any, Optional, List
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.filters import Command
from dotenv import load_dotenv
from collections import defaultdict
from flask import Flask, request, jsonify
from threading import Thread

# ==================== 🔧 Загрузка переменных окружения ====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Новые переменные для CryptoBot
CRYPTOBOT_API_KEY = os.getenv("CRYPTOBOT_API_KEY")
CRYPTOBOT_CURRENCY = os.getenv("CRYPTOBOT_CURRENCY", "USD")
CRYPTOBOT_WEBHOOK_SECRET = os.getenv("CRYPTOBOT_WEBHOOK_SECRET", "")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не задан в .env")

openai.api_key = OPENAI_API_KEY

# ==================== Flask (для вебхука и "alive" ping) ====================
app = Flask(__name__)

@app.route("/")
def home():
    return "I'm alive", 200

@app.route("/cryptobot-webhook", methods=["POST"])
def cryptobot_webhook():
    # Проверяем секрет в заголовке (настроить под реальный CryptoBot)
    header_secret = request.headers.get("X-Cryptobot-Secret") or request.headers.get("X-Webhook-Secret")
    if not CRYPTOBOT_WEBHOOK_SECRET:
        logging.warning("CRYPTOBOT_WEBHOOK_SECRET не задан — вебхук не будет валидироваться")
    else:
        if header_secret != CRYPTOBOT_WEBHOOK_SECRET:
            logging.warning("Получён вебхук с неверным секретом")
            return jsonify({"ok": False, "reason": "bad secret"}), 403

    data = request.get_json(force=True)
    try:
        order_status = data.get("status")
        payload = data.get("payload", {}) or {}
        tg_user_id = int(payload.get("tg_user_id")) if payload.get("tg_user_id") else None
        tokens = int(payload.get("tokens", 0)) if payload.get("tokens") else 0
    except Exception as e:
        logging.exception("Ошибка при разборе webhook payload: %s", e)
        return jsonify({"ok": False, "reason": "bad payload"}), 400

    if order_status == "paid" and tg_user_id and tokens > 0:
        add_tokens(tg_user_id, tokens)
        mark_made_purchase(tg_user_id)
        # Попытка отправить сообщение пользователю асинхронно
        try:
            asyncio.get_event_loop().create_task(send_payment_success_message(tg_user_id, tokens))
        except RuntimeError:
            # если цикл событий недоступен в этом потоке — запускаем в фоне
            Thread(target=lambda: asyncio.run(send_payment_success_message(tg_user_id, tokens))).start()
        return jsonify({"ok": True}), 200

    return jsonify({"ok": True}), 200


def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

Thread(target=run_flask).start()

# ==================== Telegram bot ====================
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

# Канал для бонуса
CHANNEL_USERNAME = "@Bets_OnlyForBests"

# ==================== Конфигурация экономики ====================
MODEL_COSTS = {
    "gpt-4o": 3,
    "gpt-4o-small": 2,
    "gpt-3.5-turbo": 1,
}

REFERRAL_BONUS_FUNC = lambda tokens_added: int(tokens_added)

# ==================== Работа с токенами (файл tokens.json) ====================
TOKENS_FILE = "tokens.json"

def load_tokens() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}
    try:
        with open(TOKENS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            data = json.loads(content)
            if not isinstance(data, dict):
                return {}
            migrated: Dict[str, Dict[str, Any]] = {}
            for k, v in data.items():
                key = str(k)
                if isinstance(v, dict):
                    tokens = int(v.get("tokens", 0)) if isinstance(v.get("tokens", 0), (int, float, str)) else 0
                    sub = bool(v.get("sub_bonus_given", False))
                    referrer = v.get("referrer")
                    referrals = v.get("referrals", []) if isinstance(v.get("referrals", []), list) else []
                    has_pur = bool(v.get("has_made_purchase", False))
                    migrated[key] = {
                        "tokens": tokens,
                        "sub_bonus_given": sub,
                        "referrer": referrer,
                        "referrals": referrals,
                        "has_made_purchase": has_pur
                    }
                else:
                    try:
                        tokens = int(v)
                    except Exception:
                        tokens = 0
                    migrated[key] = {
                        "tokens": tokens,
                        "sub_bonus_given": False,
                        "referrer": None,
                        "referrals": [],
                        "has_made_purchase": False
                    }
            return migrated
    except (json.JSONDecodeError, ValueError) as e:
        logging.warning(f"tokens.json повреждён или невалиден: {e}. Создаю backup и новый файл.")
        try:
            os.replace(TOKENS_FILE, TOKENS_FILE + ".bak")
            logging.info("Создан бэкап tokens.json.bak")
        except Exception as ex:
            logging.warning(f"Не удалось сделать бэкап: {ex}")
        with open(TOKENS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}
    except Exception as e:
        logging.exception(f"Неожиданная ошибка при загрузке tokens.json: {e}")
        return {}


def save_tokens(tokens: Dict[str, Dict[str, Any]]) -> None:
    tmp_file = TOKENS_FILE + ".tmp"
    try:
        serializable: Dict[str, Dict[str, Any]] = {}
        for k, v in tokens.items():
            serializable[str(k)] = {
                "tokens": int(v.get("tokens", 0)),
                "sub_bonus_given": bool(v.get("sub_bonus_given", False)),
                "referrer": v.get("referrer"),
                "referrals": v.get("referrals", []),
                "has_made_purchase": bool(v.get("has_made_purchase", False))
            }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, TOKENS_FILE)
    except Exception as e:
        logging.exception(f"Ошибка при сохранении tokens.json: {e}")

user_tokens: Dict[str, Dict[str, Any]] = load_tokens()

def _ensure_user_record(uid: str) -> None:
    if uid not in user_tokens:
        user_tokens[uid] = {
            "tokens": 0,
            "sub_bonus_given": False,
            "referrer": None,
            "referrals": [],
            "has_made_purchase": False
        }

def add_tokens(user_id: int, amount: int):
    uid = str(user_id)
    _ensure_user_record(uid)
    user_tokens[uid]["tokens"] = int(user_tokens[uid].get("tokens", 0)) + int(amount)
    save_tokens(user_tokens)
    logging.info(f"Начислено {amount} токенов пользователю {uid}. Всего токенов: {user_tokens[uid]['tokens']}")

def remove_tokens(user_id: int, amount: int) -> bool:
    uid = str(user_id)
    _ensure_user_record(uid)
    if int(user_tokens[uid].get("tokens", 0)) >= amount:
        user_tokens[uid]["tokens"] = int(user_tokens[uid].get("tokens", 0)) - amount
        save_tokens(user_tokens)
        logging.info(f"Списано {amount} токенов у {uid}. Осталось: {user_tokens[uid]['tokens']}")
        return True
    return False

def get_tokens(user_id: int) -> int:
    uid = str(user_id)
    _ensure_user_record(uid)
    return int(user_tokens[uid].get("tokens", 0))

def has_sub_bonus(user_id: int) -> bool:
    uid = str(user_id)
    _ensure_user_record(uid)
    return bool(user_tokens[uid].get("sub_bonus_given", False))

def set_sub_bonus(user_id: int) -> None:
    uid = str(user_id)
    _ensure_user_record(uid)
    user_tokens[uid]["sub_bonus_given"] = True
    save_tokens(user_tokens)

def set_referrer(user_id: int, referrer_id: Optional[int]) -> None:
    uid = str(user_id)
    _ensure_user_record(uid)
    if referrer_id is None:
        return
    ref = str(referrer_id)
    if ref == uid:
        return
    if user_tokens[uid].get("referrer"):
        return
    user_tokens[uid]["referrer"] = ref
    _ensure_user_record(ref)
    referrals = user_tokens[ref].get("referrals", [])
    if uid not in referrals:
        referrals.append(uid)
        user_tokens[ref]["referrals"] = referrals
    save_tokens(user_tokens)

def mark_made_purchase(user_id: int) -> None:
    uid = str(user_id)
    _ensure_user_record(uid)
    user_tokens[uid]["has_made_purchase"] = True
    save_tokens(user_tokens)

def user_has_made_purchase(user_id: int) -> bool:
    uid = str(user_id)
    _ensure_user_record(uid)
    return bool(user_tokens[uid].get("has_made_purchase", False))

# ==================== Память и интерфейс ====================
user_history = defaultdict(list)
feedback_stats = defaultdict(lambda: {"agree": 0, "disagree": 0})
user_model = defaultdict(lambda: "gpt-4o")
user_last_matches: Dict[int, List[str]] = {}

# ==================== Кнопки и клавиатуры ====================
def get_main_menu(user_id: int = None) -> InlineKeyboardMarkup:
    model_name = user_model[user_id] if (user_id is not None and user_id in user_model) else "gpt-4o"
    tokens = get_tokens(user_id) if user_id else 0
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Сделать прогноз", callback_data="make_forecast")],
            [InlineKeyboardButton(text="📅 Ближайшие матчи", callback_data="today_matches")],
            [InlineKeyboardButton(text="🔥 Горячие матчи дня", callback_data="hot_matches")],
            [InlineKeyboardButton(text="🕒 Моя история", callback_data="history"),
             InlineKeyboardButton(text="👤 Профиль", callback_data="profile")],
            [InlineKeyboardButton(text="📊 Фидбек", callback_data="feedback_report"),
             InlineKeyboardButton(text=f"🧾 Реферальная ссылка", callback_data="referral")],
            [InlineKeyboardButton(text=f"🧠 Модель: {model_name}", callback_data="choose_model")],
            [InlineKeyboardButton(text=f"💰 Пополнить баланс ({tokens}🔸)", callback_data="buy_tokens")],
        ])
    return kb

def get_buy_tokens_keyboard() -> InlineKeyboardMarkup:
    packages = [(5, 1.22), (10, 2.45), (30, 7.34)]
    rows = [[InlineKeyboardButton(text=f"{t}🔸 — ${price:.2f}", callback_data=f"buy_tokens:{t}:{price}")] for t, price in packages]
    return InlineKeyboardMarkup(inline_keyboard=rows)

# ==================== OpenAI wrapper ====================
def ask_openai_sync(prompt: str, model: str) -> str:
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты спортивный аналитик. Дай краткий прогноз: победитель, возможный счёт, ключевой аргумент."},
                {"role": "user", "content": f"Кому стоит отдать предпочтение в матче: {prompt}?"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI Error: {e}")
        return "⚠️ Не удалось получить прогноз."

async def ask_openai(prompt: str, model: str) -> str:
    return await asyncio.to_thread(ask_openai_sync, prompt, model)

# ==================== Получение матчей (ODDS API) ====================
async def fetch_matches_today():
    if not ODDS_API_KEY:
        return ["⚠️ ODDS_API_KEY не задан."]
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/events?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    logging.warning(f"ODDS API: статус {response.status}")
                    return ["⚠️ Не удалось загрузить матчи."]
                data = await response.json()
                if not data:
                    return ["⚠️ Сегодня нет матчей."]
                matches = []
                for event in data[:10]:
                    home = event.get('home_team', 'Home')
                    away = event.get('away_team', 'Away')
                    date = event.get('commence_time', '')[:10]
                    matches.append(f"{home} — {away} ({date})")
                return matches
    except Exception as e:
        logging.exception(f"Ошибка получения матчей: {e}")
        return ["⚠️ Ошибка при загрузке матчей."]

async def fetch_hot_matches_today():
    if not ODDS_API_KEY:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    logging.warning(f"ODDS API: статус {response.status}")
                    return []
                data = await response.json()
                matches = []
                for event in data:
                    home = event.get('home_team', 'Home')
                    away = event.get('away_team', 'Away')
                    date = event.get('commence_time', '')[:10]
                    bookmakers = event.get("bookmakers", [])
                    if not bookmakers:
                        continue
                    markets = bookmakers[0].get("markets", [])
                    if not markets:
                        continue
                    outcomes = markets[0].get("outcomes", [])
                    if len(outcomes) < 3:
                        continue
                    try:
                        od_h = float(outcomes[0].get("price", 0))
                        od_d = float(outcomes[1].get("price", 0))
                        od_a = float(outcomes[-1].get("price", 0))
                    except Exception:
                        continue
                    matches.append({
                        "home": home,
                        "away": away,
                        "date": date,
                        "odds_home": od_h,
                        "odds_draw": od_d,
                        "odds_away": od_a,
                        "max_underdog_odds": max(od_h, od_a)
                    })
                matches.sort(key=lambda x: x["max_underdog_odds"], reverse=True)
                return matches
    except Exception as e:
        logging.exception(f"Ошибка получения горячих матчей: {e}")
        return []

# ==================== Хендлеры команд и логика ====================
@dp.callback_query(F.data == "hot_matches")
async def hot_matches(callback: CallbackQuery):
    await callback.answer()
    matches = await fetch_hot_matches_today()
    if not matches:
        await callback.message.answer("⚠️ Сегодня нет горячих матчей.")
        return
    text = "🔥 *Горячие матчи дня:*\n\n"
    for i, m in enumerate(matches[:5], 1):
        text += f"{i}. {m['home']} — {m['away']} ({m['date']})\n"
        text += f"   Коэф: {m['odds_home']} / {m['odds_draw']} / {m['odds_away']}\n"
    await callback.message.answer(text, parse_mode="Markdown")

@dp.message(Command(commands=["start"]))
async def start(message: Message):
    user_id = message.from_user.id
    uid = str(user_id)

    text = (message.text or "").strip()
    m = re.match(r"^/start(?:\\s+ref_(\\d+))?$", text)
    if m:
        ref = m.group(1)
        if ref:
            set_referrer(user_id, int(ref))

    try:
        member = await bot.get_chat_member(CHANNEL_USERNAME, user_id)
        is_subscribed = member.status in ("member", "administrator", "creator")
    except Exception as e:
        logging.debug(f"Не удалось проверить подписку на канал: {e}")
        is_subscribed = False

    if uid not in user_tokens:
        _ensure_user_record(uid)
        add_tokens(user_id, 1)
        await message.answer("👋 Привет! Вам начислен 1 бесплатный токен!")

        sub_kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📢 Подписаться на канал", url=f"https://t.me/{CHANNEL_USERNAME.lstrip('@')}")],
            [InlineKeyboardButton(text="🔁 Проверить подписку и получить бонус", callback_data="check_subscription")]
        ])
        await message.answer(
            "Хотите ещё один токен? 🤩\n"
            f"Подпишитесь на наш канал {CHANNEL_USERNAME} и получите +1 токен в подарок!\n\n"
            "Нажмите <b>Проверить подписку и получить бонус</b> после подписки, чтобы бот убедился и начислил бонус.",
            reply_markup=sub_kb
        )
    elif is_subscribed and not has_sub_bonus(user_id):
        add_tokens(user_id, 1)
        set_sub_bonus(user_id)
        await message.answer("🎁 Спасибо за подписку на канал! Вам начислен 1 бонусный токен!")
    else:
        await message.answer("👋 Привет снова!")

    await message.answer(
        f"💰 Баланс: {get_tokens(user_id)} токен(ов)",
        reply_markup=get_main_menu(user_id)
    )

@dp.callback_query(F.data == "check_subscription")
async def check_subscription(callback: CallbackQuery):
    user_id = callback.from_user.id
    try:
        member = await bot.get_chat_member(CHANNEL_USERNAME, user_id)
        is_subscribed = member.status in ("member", "administrator", "creator")
    except Exception as e:
        logging.debug(f"check_subscription: {e}")
        is_subscribed = False

    if is_subscribed and not has_sub_bonus(user_id):
        add_tokens(user_id, 1)
        set_sub_bonus(user_id)
        await callback.answer("🎁 Подписка подтверждена — бонусный токен начислен!", show_alert=True)
    elif is_subscribed:
        await callback.answer("Вы уже получили бонус подписки ранее.", show_alert=True)
    else:
        await callback.answer("Вы не подписаны на канал. Подпишитесь и нажмите снова.", show_alert=True)

@dp.callback_query(F.data == "profile")
async def profile_cb(callback: CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    uid = str(user_id)
    _ensure_user_record(uid)
    data = user_tokens[uid]
    referrals = data.get("referrals", [])
    made = "Да" if data.get("has_made_purchase", False) else "Нет"
    text = (
        f"👤 <b>Профиль</b>\n\n"
        f"🔸 Токены: {data.get('tokens',0)}\n"
        f"🎁 Бонус подписки получен: {'Да' if data.get('sub_bonus_given') else 'Нет'}\n"
        f"💳 Покупки совершены: {made}\n"
        f"🤝 Реферальная ссылка: <code>/start ref_{user_id}</code>\n"
        f"👥 Приглашённые: {len(referrals)}\n"
    )
    await callback.message.answer(text, parse_mode="HTML")

# (дальше оставил остальные обработчики без изменений — они аналогичны предыдущей версии)

# Функция отправки сообщения о платеже
async def send_payment_success_message(tg_user_id: int, tokens: int):
    try:
        await bot.send_message(tg_user_id, f"✅ Оплата подтверждена. Вам зачислено {tokens} токен(ов).\n💰 Баланс: {get_tokens(tg_user_id)} токен(ов)")
    except Exception as e:
        logging.exception(f"Не удалось отправить сообщение о платеже пользователю {tg_user_id}: {e}")

# Пример создания инвойса (адаптируйте под реальный API)
async def create_crypto_invoice(tg_user_id: int, amount_usd: float, tokens: int) -> str:
    """
    Создаёт счёт в CryptoBot через HTTP API и возвращает ссылку для оплаты.
    Адаптируйте api_url/тело запроса и обработку ответа под ваш провайдер.
    """
    if not CRYPTOBOT_API_KEY:
        raise RuntimeError("CRYPTOBOT_API_KEY не задан в .env")

    payload = {
        "amount": float(f"{amount_usd:.2f}"),
        "currency": CRYPTOBOT_CURRENCY,
        "description": f"Buy {tokens} tokens for user {tg_user_id}",
        "payload": {"tg_user_id": tg_user_id, "tokens": tokens},
    }

    api_url = os.getenv("CRYPTOBOT_API_URL", "https://pay.crypt.bot/api/createInvoice")
    headers = {
        "Authorization": f"Bearer {CRYPTOBOT_API_KEY}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload, timeout=15) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                logging.error(f"Ошибка создания invoice: {resp.status} {text}")
                raise RuntimeError("Ошибка создания invoice")
            data = await resp.json()

    pay_url = data.get("pay_url") or data.get("invoice_url") or (data.get("data") or {}).get("pay_url")
    if not pay_url:
        logging.error("В ответе API не найден url оплаты: %s", data)
        raise RuntimeError("Не удалось получить ссылку на оплату из ответа API")

    return pay_url

# Запуск бота
async def main():
    logging.info("✅ Бот запущен.")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
