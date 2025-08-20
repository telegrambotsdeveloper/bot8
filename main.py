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
    LabeledPrice,
)
from aiogram.filters import Command
from dotenv import load_dotenv
from collections import defaultdict

# Рекомендуемые версии библиотек:
# - aiogram>=3.0.0
# - openai>=1.0.0
# - aiohttp>=3.8.0
# - python-dotenv>=1.0.0

# ==================== 🔧 Загрузка переменных окружения ====================
load_dotenv()

# === Админские настройки ===
SUPER_ADMIN_ID = 8185719207  # Special admin ID for token management
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BOT_USERNAME = "@MyAIChatBot1_bot"  # Username бота

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не задан в .env")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не задан в .env")

openai.api_key = OPENAI_API_KEY

# Flask для Render (оставлено без изменений)
from flask import Flask
from threading import Thread

app = Flask(__name__)

@app.route("/")
def home():
    return "I'm alive", 200

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

Thread(target=run_flask).start()

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

# Канал для бонуса
CHANNEL_USERNAME = "@Bets_OnlyForBests"

# ==================== Конфигурация экономики ====================
# Сколько звёзд = 1 токен
STARS_PER_TOKEN = 10

# Модель -> стоимость в токенах за запрос/прогноз
MODEL_COSTS = {
    "gpt-3.5-turbo": 1,  # Используем реальную модель OpenAI
}

# Бонус реферера: сколько звёзд получает пригласивший за первую покупку приглашённого
REFERRAL_BONUS_FUNC = lambda tokens_added: int(tokens_added)

# ==================== 📁 Работа с токенами (файл tokens.json) ====================
TOKENS_FILE = "tokens.json"

def load_tokens() -> Dict[str, Dict[str, Any]]:
    """
    Загружает структуру:
    { user_id: {
         "tokens": int,
         "stars": int,
         "sub_bonus_given": bool,
         "referrer": Optional[str],
         "referrals": [str],
         "has_made_purchase": bool,
         "accepted_rules": bool
       } }
    Поддерживает миграцию старого формата.
    """
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
                    # Уже в новом/частично новом формате — нормализуем поля
                    tokens = int(v.get("tokens", 0)) if isinstance(v.get("tokens", 0), (int, float, str)) else 0
                    stars = int(v.get("stars", 0)) if isinstance(v.get("stars", 0), (int, float, str)) else 0
                    sub = bool(v.get("sub_bonus_given", False))
                    referrer = v.get("referrer")
                    referrals = v.get("referrals", []) if isinstance(v.get("referrals", []), list) else []
                    has_pur = bool(v.get("has_made_purchase", False))
                    accepted_rules = bool(v.get("accepted_rules", False))
                    migrated[key] = {
                        "tokens": tokens,
                        "stars": stars,
                        "sub_bonus_given": sub,
                        "referrer": referrer,
                        "referrals": referrals,
                        "has_made_purchase": has_pur,
                        "accepted_rules": accepted_rules
                    }
                else:
                    # Старый формат: просто число токенов
                    try:
                        tokens = int(v)
                    except Exception:
                        tokens = 0
                    migrated[key] = {
                        "tokens": tokens,
                        "stars": 0,
                        "sub_bonus_given": False,
                        "referrer": None,
                        "referrals": [],
                        "has_made_purchase": False,
                        "accepted_rules": False
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
                "stars": int(v.get("stars", 0)),
                "sub_bonus_given": bool(v.get("sub_bonus_given", False)),
                "referrer": v.get("referrer"),
                "referrals": v.get("referrals", []),
                "has_made_purchase": bool(v.get("has_made_purchase", False)),
                "accepted_rules": bool(v.get("accepted_rules", False))
            }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, TOKENS_FILE)
    except Exception as e:
        logging.exception(f"Ошибка при сохранении tokens.json: {e}")

# Загружаем
user_tokens: Dict[str, Dict[str, Any]] = load_tokens()

# Вспомогательные операции
def _ensure_user_record(uid: str) -> None:
    if uid not in user_tokens:
        user_tokens[uid] = {
            "tokens": 2,  # Новые пользователи получают 5 токенов
            "stars": 0,
            "sub_bonus_given": False,
            "referrer": None,
            "referrals": [],
            "has_made_purchase": False,
            "accepted_rules": False
        }

def add_stars(user_id: int, amount: int):
    uid = str(user_id)
    _ensure_user_record(uid)
    user_tokens[uid]["stars"] = int(user_tokens[uid].get("stars", 0)) + int(amount)
    save_tokens(user_tokens)
    logging.info(f"Начислено {amount}⭐ пользователю {uid}. Всего ⭐: {user_tokens[uid]['stars']}")

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

def get_stars(user_id: int) -> int:
    uid = str(user_id)
    _ensure_user_record(uid)
    return int(user_tokens[uid].get("stars", 0))

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

def has_accepted_rules(user_id: int) -> bool:
    uid = str(user_id)
    _ensure_user_record(uid)
    return bool(user_tokens[uid].get("accepted_rules", False))

def set_accepted_rules(user_id: int) -> None:
    uid = str(user_id)
    _ensure_user_record(uid)
    user_tokens[uid]["accepted_rules"] = True
    save_tokens(user_tokens)

# ==================== 🗂 Память и настройки ====================
user_history = defaultdict(list)  # key — int user_id
user_last_matches: Dict[int, List[str]] = {}

# ==================== 🔘 Кнопки и клавиатуры ====================
def get_main_menu(user_id: int = None) -> InlineKeyboardMarkup:
    tokens = get_tokens(user_id) if user_id else 0
    stars = get_stars(user_id) if user_id else 0
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Сделать прогноз", callback_data="make_forecast")],
            [InlineKeyboardButton(text="📅 Ближайшие матчи", callback_data="today_matches")],
            [InlineKeyboardButton(text="👤 Профиль", callback_data="profile")],
            [InlineKeyboardButton(text="🧾 Реферальная ссылка", callback_data="referral")],
            [InlineKeyboardButton(text="ℹ Как это работает", callback_data="how_it_works")],
            [InlineKeyboardButton(text="💰 Пополнить баланс", callback_data="buy_stars")],
            [InlineKeyboardButton(text="⚠️ Мы против азартных игр", callback_data="anti_gambling")]
        ])
    if user_id == SUPER_ADMIN_ID:
        kb.inline_keyboard.append([InlineKeyboardButton(text="🔧 Управление токенами", callback_data="super_admin_token_management")])
    return kb

def get_buy_stars_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="50⭐", callback_data="buy_stars:50")],
            [InlineKeyboardButton(text="100⭐", callback_data="buy_stars:100")],
            [InlineKeyboardButton(text="300⭐", callback_data="buy_stars:300")],
        ]
    )

def get_rules_acceptance_keyboard(stars: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Согласен с правилами", callback_data=f"accept_rules:{stars}")],
            [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")]
        ]
    )

def get_super_admin_token_management_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="👤 Изменить токены пользователя", callback_data="super_admin_change_tokens")]
        ]
    )

# ==================== 🌍 Перевод команд (пример) ====================
team_translation = {
    "Manchester City": "Манчестер Сити",
    "Liverpool": "Ливерпуль",
    "Barcelona": "Барселона",
    "Real Madrid": "Реал",
    "Chelsea": "Челси",
    "Arsenal": "Арсенал",
}

def translate_team(name: str) -> str:
    return team_translation.get(name, name)

# ==================== 🧠 OpenAI — обёртка ====================
def ask_openai_sync(prompt: str, model: str = "gpt-3.5-turbo") -> str:
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
        logging.error(f"OpenAI Error details: {type(e).__name__}: {str(e)}")
        return "⚠️ Не удалось получить прогноз. Проверьте API-ключ или лимиты."

async def ask_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    return await asyncio.to_thread(ask_openai_sync, prompt, model)

# ==================== 📅 Получение матчей (ODDS API) ====================
async def fetch_matches_today():
    if not ODDS_API_KEY:
        logging.error("ODDS_API_KEY не задан в .env")
        return ["⚠️ ODDS_API_KEY не задан."]
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/events?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                logging.info(f"ODDS API response status: {response.status}")
                if response.status != 200:
                    logging.warning(f"ODDS API: статус {response.status}")
                    return ["⚠️ Не удалось загрузить матчи."]
                data = await response.json()
                logging.info(f"ODDS API response data length: {len(data)}")
                if not data:
                    return ["⚠️ Сегодня нет матчей в английской Премьер-лиге."]
                matches = []
                for event in data[:10]:
                    home = translate_team(event.get('home_team', 'Home'))
                    away = translate_team(event.get('away_team', 'Away'))
                    date = event.get('commence_time', '')[:10]
                    matches.append(f"{home} — {away} ({date})")
                return matches
    except aiohttp.ClientError as e:
        logging.exception(f"Сетевая ошибка при получении матчей: {e}")
        return ["⚠️ Ошибка сети при загрузке матчей."]
    except asyncio.TimeoutError:
        logging.exception("Тайм-аут при запросе к ODDS API")
        return ["⚠️ Тайм-аут при загрузке матчей."]
    except Exception as e:
        logging.exception(f"Неожиданная ошибка при получении матчей: {e}")
        return ["⚠️ Ошибка при загрузке матчей."]

# ==================== 📍 Хендлеры команд ====================

@dp.message(Command(commands=["start"]))
async def start(message: Message):
    user_id = message.from_user.id
    uid = str(user_id)

    # Check for referral in /start parameters
    text = (message.text or "").strip()
    m = re.match(r"^/start(?:\s+(\d+))?$", text)
    if m:
        ref = m.group(1)
        if ref:
            set_referrer(user_id, int(ref))

    # Проверка подписки
    try:
        member = await bot.get_chat_member(CHANNEL_USERNAME, user_id)
        is_subscribed = member.status in ("member", "administrator", "creator")
    except Exception as e:
        logging.debug(f"Не удалось проверить подписку на канал: {e}")
        is_subscribed = False

    sub_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📢 Подписаться на канал", url=f"https://t.me/{CHANNEL_USERNAME.lstrip('@')}")],
        [InlineKeyboardButton(text="🔁 Проверить подписку и получить бонус", callback_data="check_subscription")]
    ])

    if uid not in user_tokens:
        _ensure_user_record(uid)
        text = "👋 Привет! Вам начислено 2 бесплатных токена!\n\n"
        text += "Хотите ещё один токен? 🤩\n"
        text += f"Подпишитесь на наш канал {CHANNEL_USERNAME} и получите +1 токен в подарок!\n\n"
        text += "Нажмите <b>Проверить подписку и получить бонус</b> после подписки, чтобы бот убедился и начислил бонус.\n\n"
    elif is_subscribed and not has_sub_bonus(user_id):
        add_tokens(user_id, 1)
        set_sub_bonus(user_id)
        text = "🎁 Спасибо за подписку на канал! Вам начислен 1 бонусный токен!\n\n"
    else:
        text = "👋 Привет снова!\n\n"

    text += (
        f"💰 Баланс: {get_tokens(user_id)} токен(ов) • {get_stars(user_id)}⭐\n\n"
        f"⚠️ Мы против азартных игр и ставок. Наш сервис предназначен только для аналитики и прогнозирования спортивных событий. Мы не гарантируем правильность прогнозов. Не воспринимайте наши прогнозы как проверенная информация, это лишь предположения"
    )
    await message.answer(text, reply_markup=get_main_menu(user_id))

# Кнопка проверки подписки
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
        await callback.message.answer("🎁 Подписка подтверждена — бонусный токен начислен!")
        await callback.answer()
    elif is_subscribed:
        await callback.answer("Вы уже получили бонус подписки ранее.", show_alert=True)
    else:
        await callback.answer("Вы не подписаны на канал. Подпишитесь и нажмите снова.", show_alert=True)

# Профиль
@dp.callback_query(F.data == "profile")
async def profile_cb(callback: CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    uid = str(user_id)
    _ensure_user_record(uid)
    data = user_tokens[uid]
    referrals = data.get("referrals", [])
    made = "Да" if data.get("has_made_purchase", False) else "Нет"
    accepted = "Да" if data.get("accepted_rules", False) else "Нет"
    text = (
        f"👤 <b>Профиль</b>\n\n"
        f"🔸 Токены: {data.get('tokens',0)}\n"
        f"⭐ Звёзды: {data.get('stars',0)}\n"
        f"🎁 Бонус подписки получен: {'Да' if data.get('sub_bonus_given') else 'Нет'}\n"
        f"💳 Покупки совершены: {made}\n"
        f"✅ Согласен с правилами: {accepted}\n"
        f"🤝 Реферальная ссылка: <code>https://t.me/{BOT_USERNAME}?start={user_id}</code>\n"
        f"👥 Приглашённые: {len(referrals)}\n"
    )
    await callback.message.answer(text, parse_mode="HTML")

@dp.callback_query(F.data == "referral")
async def referral_cb(callback: CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    uid = str(user_id)
    _ensure_user_record(uid)
    referrals = user_tokens[uid].get("referrals", [])

    text = (
        "🤝 <b>Реферальная программа</b>\n\n"
        "Приглашайте друзей и получайте звёзды за их первую покупку.\n"
        f"Ваша ссылка для приглашений: <code>https://t.me/{BOT_USERNAME}?start={user_id}</code>\n"
        "Отправьте её друзьям или разместите в соцсетях.\n\n"
        "📌 Как это работает:\n"
        "— Человек заходит в бота по вашей ссылке.\n"
        "— Делает первую покупку.\n"
        "— Вы получаете бонус в звёздах (автоконвертация в токены)."
    )

    if not referrals:
        text += "\n\n👥 У вас пока нет приглашённых."
    else:
        text += "\n\n👥 Ваши приглашённые:\n"
        for r in referrals:
            tokens_r = user_tokens.get(r, {}).get("tokens", 0)
            stars_r = user_tokens.get(r, {}).get("stars", 0)
            made = user_tokens.get(r, {}).get("has_made_purchase", False)
            text += f"• Пользователь {r} — Покупал: {'Да' if made else 'Нет'} — {tokens_r}🔸 / {stars_r}⭐\n"

    await callback.message.answer(text, parse_mode="HTML")

@dp.callback_query(F.data == "anti_gambling")
async def anti_gambling(callback: CallbackQuery):
    await callback.answer()
    text = (
        "⚠️ <b>Мы против азартных игр</b>\n\n"
        "Наш сервис предназначен исключительно для анализа и прогнозирования спортивных событий. "
        "Мы не поддерживаем и не поощряем ставки или любые формы азартных игр. "
        "Пожалуйста, используйте наши прогнозы только для информационных целей и развлечений. "
        "Мы не гарантируем правильность прогнозов. Не воспринимайте наши прогнозы как проверенная информация, это лишь предположения. "
        "Возврат средств невозможен и не предусмотрен. "
    )
    await callback.message.answer(text, parse_mode="HTML")

# Покупка звёзд — показ клавиатуры
@dp.callback_query(F.data == "buy_stars")
async def buy_stars_menu(callback: CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    if not has_accepted_rules(user_id):
        await callback.message.answer(
            "⚠️ Для покупки звёзд вы должны согласиться с нашими правилами:\n\n"
            "Мы против азартных игр. Наш сервис предназначен только для анализа и прогнозирования спортивных событий. "
            "Используйте прогнозы только для информационных целей. Возврат средств невозможен и не предусмотрен.\n\n"
            "Согласны ли вы с этими правилами?",
            reply_markup=get_rules_acceptance_keyboard(0)
        )
    else:
        await callback.message.answer("Выберите пакет звёзд (покупается в XTR):", reply_markup=get_buy_stars_keyboard())

# Подтверждение принятия правил
@dp.callback_query(F.data.startswith("accept_rules:"))
async def accept_rules(callback: CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    stars = int(callback.data.split(":", 1)[1])
    set_accepted_rules(user_id)
    if stars > 0:
        await create_invoice_for_stars(callback, stars)
    else:
        await callback.message.answer("✅ Правила приняты! Теперь вы можете приобрести звёзды.", reply_markup=get_buy_stars_keyboard())

# Создание инвойса для покупки звёзд
@dp.callback_query(F.data.startswith("buy_stars:"))
async def create_invoice(callback: CallbackQuery):
    user_id = callback.from_user.id
    if not has_accepted_rules(user_id):
        await callback.message.answer(
            "⚠️ Для покупки звёзд вы должны согласиться с нашими правилами:\n\n"
            "Мы против азартных игр. Наш сервис предназначен только для анализа и прогнозирования спортивных событий. "
            "Используйте прогнозы только для информационных целей.\n\n"
            "Согласны ли вы с этими правилами?",
            reply_markup=get_rules_acceptance_keyboard(int(callback.data.split(":", 1)[1]))
        )
        await callback.answer()
        return

    try:
        stars = int(callback.data.split(":", 1)[1])
    except Exception:
        await callback.answer("Ошибка пакета.", show_alert=True)
        return
    await create_invoice_for_stars(callback, stars)

async def create_invoice_for_stars(callback: CallbackQuery, stars: int):
    prices = [LabeledPrice(label=f"{stars}⭐", amount=stars)]
    try:
        await bot.send_invoice(
            chat_id=callback.message.chat.id,
            title="Пополнение баланса звёзд",
            description=f"Пакет: {stars}⭐",
            payload=f"buy_stars_{stars}",
            provider_token="",  # empty for XTR
            currency="XTR",
            prices=prices,
            start_parameter=f"buystars_{stars}"
        )
        await callback.answer()
    except Exception as e:
        logging.exception(f"Ошибка при отправке invoice: {e}")
        await callback.answer("Не удалось создать счёт. Попробуйте позже.", show_alert=True)

@dp.pre_checkout_query()
async def process_pre_checkout_query(pre_checkout_query):
    try:
        await bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)
    except Exception as e:
        logging.exception(f"pre_checkout error: {e}")

# Обработка успешной оплаты
@dp.message(F.successful_payment)
async def successful_payment(message: Message):
    payload = message.successful_payment.invoice_payload
    user_id = message.from_user.id
    if not payload:
        return
    if payload.startswith("buy_stars_"):
        try:
            stars = int(payload.split("_")[2])
        except Exception:
            await message.answer("Оплата прошла, но не удалось распознать пакет. Свяжитесь с поддержкой.")
            return

        # Начисляем звёзды
        add_stars(user_id, stars)

        # Конвертация: каждые STARS_PER_TOKEN звёзд -> 1 токен
        uid = str(user_id)
        _ensure_user_record(uid)
        current_stars = int(user_tokens[uid].get("stars", 0))
        tokens_to_add = current_stars // STARS_PER_TOKEN
        remainder_stars = current_stars % STARS_PER_TOKEN

        if tokens_to_add > 0:
            add_tokens(user_id, tokens_to_add)
            # Обновляем остаток звёзд
            user_tokens[uid]["stars"] = remainder_stars
            save_tokens(user_tokens)

        # Реферальная логика
        ref = user_tokens[uid].get("referrer")
        if ref and not user_tokens[uid].get("has_made_purchase", False):
            inviter = ref
            inviter_bonus = REFERRAL_BONUS_FUNC(tokens_to_add)
            if inviter_bonus > 0:
                add_stars(int(inviter), inviter_bonus)
            # Пометить, что приглашённый совершил покупку
            mark_made_purchase(user_id)

        # Сообщение пользователю
        await message.answer(
            f"✅ Оплата принята. Вам зачислено {stars}⭐.\n"
            f"↪ Автоконвертация: {tokens_to_add} токенов (осталось {remainder_stars}⭐ на балансе).\n"
            f"💰 Баланс: {get_tokens(user_id)} токен(ов), {get_stars(user_id)}⭐"
        )

# Супер-админ панель для управления токенами
@dp.callback_query(F.data == "super_admin_token_management")
async def super_admin_token_management(callback: CallbackQuery):
    if callback.from_user.id != SUPER_ADMIN_ID:
        await callback.answer("⛔ Доступ запрещён.", show_alert=True)
        return
    await callback.answer()
    await callback.message.answer("🔧 Управление токенами пользователей:", reply_markup=get_super_admin_token_management_keyboard())

@dp.callback_query(F.data == "super_admin_change_tokens")
async def super_admin_change_tokens(callback: CallbackQuery):
    if callback.from_user.id != SUPER_ADMIN_ID:
        await callback.answer("⛔ Доступ запрещён.", show_alert=True)
        return
    await callback.answer()
    await callback.message.answer("Введите команду в формате: /set_tokens <user_id> <amount>\nПример: /set_tokens 123456789 10")

@dp.message(Command(commands=["set_tokens"]))
async def set_tokens_command(message: Message):
    if message.from_user.id != SUPER_ADMIN_ID:
        await message.answer("⛔ Доступ запрещён.")
        return
    args = message.text.strip().split(maxsplit=2)
    if len(args) < 3:
        await message.answer("❌ Укажите user_id и количество токенов: /set_tokens <user_id> <amount>")
        return
    try:
        target_user_id = int(args[1])
        amount = int(args[2])
        if amount < 0:
            await message.answer("❌ Количество токенов не может быть отрицательным.")
            return
        uid = str(target_user_id)
        _ensure_user_record(uid)
        user_tokens[uid]["tokens"] = amount
        save_tokens(user_tokens)
        logging.info(f"Супер-админ {SUPER_ADMIN_ID} установил {amount} токенов для пользователя {uid}")
        await message.answer(f"✅ Пользователю {target_user_id} установлено {amount} токенов.")
    except ValueError:
        await message.answer("❌ Неверный формат user_id или количества токенов.")

# ==================== Predict (центральная логика) ====================
@dp.message(Command(commands=["predict"]))
async def predict(message: Message):
    user_id = message.from_user.id
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer("❌ Укажите матч в формате: /predict Команда1 - Команда2")
        return

    match_text = args[1].strip()
    if not re.match(r"^.+\s(-|—|vs)\s.+$", match_text, re.IGNORECASE):
        await message.answer("❌ Формат неверный. Используйте: Команда1 - Команда2 или Команда1 vs Команда2")
        return

    # Стоимость модели (в токенах)
    model = "gpt-3.5-turbo"
    cost = MODEL_COSTS.get(model, 1)

    if get_tokens(user_id) < cost:
        await message.answer(f"❌ У вас недостаточно токенов. Для прогноза требуется {cost} токен(ов). Купите звёзды и конвертируйте их в токены.", reply_markup=get_buy_stars_keyboard())
        return

    # Списываем токены
    if not remove_tokens(user_id, cost):
        await message.answer("❌ Не удалось списать токены. Попробуйте ещё раз.")
        return

    # Сохраняем в историю
    user_history[user_id].append(match_text)
    if len(user_history[user_id]) > 200:
        user_history[user_id] = user_history[user_id][-200:]

    await message.answer("🤖 Генерирую прогноз...")
    forecast = await ask_openai(match_text, model)

    await message.answer(
        f"📊 *Прогноз* (стоимость {cost} токенов):\n{forecast}\n\n"
        f"💰 Остаток токенов: {get_tokens(user_id)}",
        parse_mode="Markdown"
    )

# ==================== Остальные коллбэки ====================
@dp.callback_query(F.data == "today_matches")
async def today_matches(callback: CallbackQuery):
    await callback.answer()
    matches = await fetch_matches_today()
    # Сохраняем для пользователя
    user_last_matches[callback.from_user.id] = matches
    text = "📅 *Ближайшие матчи:*\n\n" + "\n".join(f"{i+1}. {m}" for i, m in enumerate(matches))
    # Добавляем кнопку сделать прогноз
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Сделать прогноз по матчу", callback_data="choose_match")],
    ])
    await callback.message.answer(text, parse_mode="Markdown", reply_markup=kb)

@dp.callback_query(F.data == "choose_match")
async def choose_match(callback: CallbackQuery):
    user_id = callback.from_user.id
    matches = user_last_matches.get(user_id)
    if not matches:
        await callback.answer("Сначала загрузите список матчей через 'Ближайшие матчи'.", show_alert=True)
        return
    # Создаём кнопки списка матчей
    kb_rows = []
    for idx, m in enumerate(matches[:10]):
        kb_rows.append([InlineKeyboardButton(text=f"{idx+1}. {m.split('(')[0].strip()}", callback_data=f"select_match:{idx}")])
    kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)
    await callback.message.answer("Выберите матч для прогноза:", reply_markup=kb)
    await callback.answer()

@dp.callback_query(F.data.startswith("select_match:"))
async def select_match(callback: CallbackQuery):
    user_id = callback.from_user.id
    idx = int(callback.data.split(":", 1)[1])
    matches = user_last_matches.get(user_id, [])
    if idx < 0 or idx >= len(matches):
        await callback.answer("Неправильный матч.", show_alert=True)
        return
    match_text = matches[idx]
    model = "gpt-3.5-turbo"
    cost = MODEL_COSTS.get(model, 1)

    # Подтверждение
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"Подтвердить (спишется {cost} токенов)", callback_data=f"confirm_forecast:{idx}")],
        [InlineKeyboardButton(text="Отмена", callback_data="cancel")],
    ])
    await callback.message.answer(f"Вы выбрали:\n<b>{match_text}</b>\n\nМодель: {model}\nСтоимость: {cost} токен(ов).", parse_mode="HTML", reply_markup=kb)
    await callback.answer()

@dp.callback_query(F.data.startswith("confirm_forecast:"))
async def confirm_forecast(callback: CallbackQuery):
    user_id = callback.from_user.id
    idx = int(callback.data.split(":", 1)[1])
    matches = user_last_matches.get(user_id, [])
    if idx < 0 or idx >= len(matches):
        await callback.answer("Неправильный матч.", show_alert=True)
        return
    match_text = matches[idx]
    model = "gpt-3.5-turbo"
    cost = MODEL_COSTS.get(model, 1)

    if get_tokens(user_id) < cost:
        await callback.answer("У вас недостаточно токенов для прогноза.", show_alert=True)
        return

    # Списываем и генерируем
    if not remove_tokens(user_id, cost):
        await callback.answer("Не удалось списать токены.", show_alert=True)
        return

    await callback.message.answer("🤖 Генерирую прогноз...")
    forecast = await ask_openai(match_text, model)

    # Лог истории
    user_history[user_id].append(f"{match_text}")
    if len(user_history[user_id]) > 200:
        user_history[user_id] = user_history[user_id][-200:]

    await callback.message.answer(
        f"📊 *Прогноз*:\n{forecast}\n\n"
        f"💰 Остаток токенов: {get_tokens(user_id)}",
        parse_mode="Markdown"
    )
    await callback.answer()

@dp.callback_query(F.data == "cancel")
async def cancel_cb(callback: CallbackQuery):
    await callback.answer("Отменено.", show_alert=False)

@dp.callback_query(F.data == "make_forecast")
async def make_forecast(callback: CallbackQuery):
    await callback.answer()
    await callback.message.answer(
        "📊 *Как сделать прогноз:*\n\n"
        "— Можно использовать кнопку *Ближайшие матчи* и выбрать конкретный матч.\n"
        "— Или отправить команду:\n"
        "/predict Команда1 - Команда2\n\n"
        f"⚠️ Стоимость прогноза: 1 токен.\n"
        f"10⭐ = 1 токен. Пополнить баланс: нажмите «Пополнить баланс» в меню.",
        parse_mode="Markdown"
    )

@dp.callback_query(F.data == "how_it_works")
async def how_it_works(callback: CallbackQuery):
    await callback.answer()
    text = (
        "ℹ <b>Как это работает</b>\n\n"
        "1️⃣ При входе вы получаете 5 токенов бесплатно.\n"
        "2️⃣ Подпишитесь на наш канал — получите ещё 1 токен.\n"
        "3️⃣ Токены тратите на прогнозы спортивных матчей.\n"
        "4️⃣ Прогнозы генерируются с помощью искусственного интеллекта.\n\n"
        "💡 Пример:\n"
        "Вы выбираете матч <i>Барселона — Реал</i>.\n"
        "Бот анализирует статистику и даёт прогноз: победитель, возможный счёт, аргументы.\n"
        "Стоимость прогноза: 1 токен.\n\n"
        "⚠️ Мы против азартных игр и ставок. Наш сервис предназначен только для аналитики и прогнозирования спортивных событий."
    )
    await callback.message.answer(text, parse_mode="HTML")

# ==================== ▶️ Запуск бота ====================
async def main():
    logging.info("✅ Бот запущен.")
    logging.info(f"TELEGRAM_TOKEN: {TELEGRAM_TOKEN[:10]}...")
    logging.info(f"OPENAI_API_KEY: {OPENAI_API_KEY[:5]}...")
    try:
        bot_info = await bot.get_me()
        logging.info(f"Bot info: {bot_info}")
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Ошибка при запуске бота: {e}")
        raise
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())

