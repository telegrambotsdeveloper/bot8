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

# ==================== 🔧 Загрузка переменных окружения ====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не задан в .env")

openai.api_key = OPENAI_API_KEY
import os
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
# сколько звёзд = 1 токен
STARS_PER_TOKEN = 10

# модель -> стоимость в токенах за запрос/прогноз
MODEL_COSTS = {
    "gpt-4o": 3,           # дорогая модель — списывает 3 токена
    "gpt-4o-small": 2,     # пример
    "gpt-3.5-turbo": 1,    # дешёвая модель — 1 токен
}

# бонус реферера: сколько звёзд получает пригласивший за первую покупку приглашённого
# (в твоём примере: пригласивший получил бы 5 звёзд, если приглашённый купил 5 токенов за 50⭐)
# мы реализуем: inviter_bonus_stars = tokens_added (можно изменить)
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
         "has_made_purchase": bool
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
                    # уже в новом/частично новом формате — нормализуем поля
                    tokens = int(v.get("tokens", 0)) if isinstance(v.get("tokens", 0), (int, float, str)) else 0
                    stars = int(v.get("stars", 0)) if isinstance(v.get("stars", 0), (int, float, str)) else 0
                    sub = bool(v.get("sub_bonus_given", False))
                    referrer = v.get("referrer")
                    referrals = v.get("referrals", []) if isinstance(v.get("referrals", []), list) else []
                    has_pur = bool(v.get("has_made_purchase", False))
                    migrated[key] = {
                        "tokens": tokens,
                        "stars": stars,
                        "sub_bonus_given": sub,
                        "referrer": referrer,
                        "referrals": referrals,
                        "has_made_purchase": has_pur
                    }
                else:
                    # старый формат: просто число токенов
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
                "stars": int(v.get("stars", 0)),
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

# загружаем
user_tokens: Dict[str, Dict[str, Any]] = load_tokens()

# вспомогательные операции
def _ensure_user_record(uid: str) -> None:
    if uid not in user_tokens:
        user_tokens[uid] = {
            "tokens": 0,
            "stars": 0,
            "sub_bonus_given": False,
            "referrer": None,
            "referrals": [],
            "has_made_purchase": False
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
    # не позволяем самому себя как реферера
    if ref == uid:
        return
    # если уже есть реферер — не перезаписываем
    if user_tokens[uid].get("referrer"):
        return
    user_tokens[uid]["referrer"] = ref
    # добавляем в список рефералов у пригласившего
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

# ==================== 🗂 Память и настройки ====================
user_history = defaultdict(list)  # key — int user_id
feedback_stats = defaultdict(lambda: {"agree": 0, "disagree": 0})
user_model = defaultdict(lambda: "gpt-4o")  # default model per user (keys are int user_id)

# Для матче-взаимодействия держим временный список матчей на пользователя
user_last_matches: Dict[int, List[str]] = {}

# ==================== 🔘 Кнопки и клавиатуры ====================
def get_main_menu(user_id: int = None) -> InlineKeyboardMarkup:
    model_name = user_model[user_id] if (user_id is not None and user_id in user_model) else "gpt-4o"
    tokens = get_tokens(user_id) if user_id else 0
    stars = get_stars(user_id) if user_id else 0
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Сделать прогноз", callback_data="make_forecast")],
            [InlineKeyboardButton(text="📅 Ближайшие матчи", callback_data="today_matches")],
            [InlineKeyboardButton(text="🔥 Горячие матчи дня", callback_data="hot_matches")],
            [InlineKeyboardButton(text="🕒 Моя история", callback_data="history"),
             InlineKeyboardButton(text="👤 Профиль", callback_data="profile")],
            [InlineKeyboardButton(text="📊 Фидбек", callback_data="feedback_report"),
             InlineKeyboardButton(text=f"🧾 Реферальная ссылка", callback_data="referral")],
            [InlineKeyboardButton(text="ℹ Как это работает", callback_data="how_it_works")],
            [InlineKeyboardButton(text=f"🧠 Модель: {model_name}", callback_data="choose_model")],
            [InlineKeyboardButton(text=f"💰 Пополнить баланс ({stars}⭐ / {tokens}🔸)", callback_data="buy_stars")],
        ])
    return kb

def get_buy_stars_keyboard() -> InlineKeyboardMarkup:
    # Пользователь покупает звёзды (stars), а не токены.
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="50⭐", callback_data="buy_stars:50")],
            [InlineKeyboardButton(text="100⭐", callback_data="buy_stars:100")],
            [InlineKeyboardButton(text="300⭐", callback_data="buy_stars:300")],
        ]
    )

def get_feedback_buttons(match: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👍", callback_data=f"agree:{match}"),
         InlineKeyboardButton(text="👎", callback_data=f"disagree:{match}")]
    ])


def get_model_choice_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⚡ GPT-4o", callback_data="model:gpt-4o"),
            InlineKeyboardButton(text="✨ GPT-4o-mini", callback_data="model:gpt-4o-mini"),
        ],
        [
            InlineKeyboardButton(text="💡 GPT-3.5", callback_data="model:gpt-3.5-turbo")
        ]
    ])
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

# ==================== 📅 Получение матчей (ODDS API) ====================
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
                    home = translate_team(event.get('home_team', 'Home'))
                    away = translate_team(event.get('away_team', 'Away'))
                    date = event.get('commence_time', '')[:10]
                    matches.append(f"{home} — {away} ({date})")
                return matches
    except Exception as e:
        logging.exception(f"Ошибка получения матчей: {e}")
        return ["⚠️ Ошибка при загрузке матчей."]


# ==================== Горячие матчи (реальные коэффициенты) ====================
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
                    home = translate_team(event.get('home_team', 'Home'))
                    away = translate_team(event.get('away_team', 'Away'))
                    date = event.get('commence_time', '')[:10]
                    bookmakers = event.get("bookmakers", [])
                    if not bookmakers:
                        continue
                    markets = bookmakers[0].get("markets", [])
                    if not markets:
                        continue
                    outcomes = markets[0].get("outcomes", [])
                    # find three outcomes (home/draw/away) — ensure prices are numbers
                    if len(outcomes) < 3:
                        continue
                    # normalize by mapping by name if order unpredictable
                    price_map = {o.get("name"): o.get("price") for o in outcomes}
                    # try fallback to first three if mapping not standard
                    try:
                        odds_home = price_map.get(event.get('home_team')) or outcomes[0].get("price")
                    except Exception:
                        odds_home = outcomes[0].get("price")
                    try:
                        odds_away = price_map.get(event.get('away_team')) or outcomes[-1].get("price")
                    except Exception:
                        odds_away = outcomes[-1].get("price")
                    try:
                        odds_draw = price_map.get("Draw") or outcomes[1].get("price")
                    except Exception:
                        odds_draw = outcomes[1].get("price")
                    # ensure floats
                    try:
                        od_h = float(odds_home)
                        od_d = float(odds_draw)
                        od_a = float(odds_away)
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

# ==================== 📍 Хендлеры команд ====================

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

    # Check for referral in /start parameters
    # aiogram passes full text in message.text: "/start" or "/start ref_12345"
    # If ref present, set referrer
    text = (message.text or "").strip()
    m = re.match(r"^/start(?:\s+ref_(\d+))?$", text)
    if m:
        ref = m.group(1)
        if ref:
            set_referrer(user_id, int(ref))

    # Проверка подписки (пытаемся, но не ломаем работу, если ошибка)
    try:
        member = await bot.get_chat_member(CHANNEL_USERNAME, user_id)
        is_subscribed = member.status in ("member", "administrator", "creator")
    except Exception as e:
        logging.debug(f"Не удалось проверить подписку на канал: {e}")
        is_subscribed = False

    # Новый пользователь: создаём запись и даём 1 токен + предложение подписаться
    if uid not in user_tokens:
        _ensure_user_record(uid)
        # выдаём 1 токен бесплатно
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
    # Повторный запуск: если подписан и бонус ещё не давали — начисляем +1 и помечаем
    elif is_subscribed and not has_sub_bonus(user_id):
        add_tokens(user_id, 1)
        set_sub_bonus(user_id)
        await message.answer("🎁 Спасибо за подписку на канал! Вам начислен 1 бонусный токен!")

    else:
        await message.answer("👋 Привет снова!")

    await message.answer(
        f"💰 Баланс: {get_tokens(user_id)} токен(ов) • {get_stars(user_id)}⭐",
        reply_markup=get_main_menu(user_id)
    )

# Кнопка проверки подписки (при необходимости)
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
    text = (
        f"👤 <b>Профиль</b>\n\n"
        f"🔸 Токены: {data.get('tokens',0)}\n"
        f"⭐ Звёзды: {data.get('stars',0)}\n"
        f"🎁 Бонус подписки получен: {'Да' if data.get('sub_bonus_given') else 'Нет'}\n"
        f"💳 Покупки совершены: {made}\n"
        f"🤝 Реферальная ссылка: <code>/start ref_{user_id}</code>\n"
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
        f"Ваша ссылка для приглашений: <code>/start ref_{user_id}</code>\n"
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

@dp.callback_query(F.data == "referral")
async def referral_cb(callback: CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    uid = str(user_id)
    _ensure_user_record(uid)
    referrals = user_tokens[uid].get("referrals", [])

    text = (
        "🤝 <b>Реферальная программа</b>\n\n"
        "Приглашайте друзей и получайте звёзды за их первой покупку.\n"
        f"Ваша ссылка для приглашений: <code>/start ref_{user_id}</code>\n"
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

@dp.message(Command(commands=["stats"]))
async def stats(message: Message):
    await message.answer(f"💰 У вас {get_tokens(message.from_user.id)} токен(ов) и {get_stars(message.from_user.id)}⭐")

# Покупка звёзд — показ клавиатуры
@dp.callback_query(F.data == "buy_stars")
async def buy_stars_menu(callback: CallbackQuery):
    await callback.answer()
    await callback.message.answer("Выберите пакет звёзд (покупается в XTR):", reply_markup=get_buy_stars_keyboard())

# Создание инвойса для покупки звёзд
@dp.callback_query(F.data.startswith("buy_stars:"))
async def create_invoice(callback: CallbackQuery):
    try:
        stars = int(callback.data.split(":", 1)[1])
    except Exception:
        await callback.answer("Ошибка пакета.", show_alert=True)
        return

    # amount in XTR is passed as integer number of stars (no *100)
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

# Обработка успешной оплаты: начисляем звезды, конвертим в токены, начисляем рефереру бонус если нужно
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

        # начисляем звезды
        add_stars(user_id, stars)

        # конвертация: каждые STARS_PER_TOKEN звёзд -> 1 токен
        uid = str(user_id)
        _ensure_user_record(uid)
        current_stars = int(user_tokens[uid].get("stars", 0))
        tokens_to_add = current_stars // STARS_PER_TOKEN
        remainder_stars = current_stars % STARS_PER_TOKEN

        if tokens_to_add > 0:
            add_tokens(user_id, tokens_to_add)
            # обновляем остаток звёзд
            user_tokens[uid]["stars"] = remainder_stars
            save_tokens(user_tokens)

        # реферальная логика: если у пользователя есть реферер и он ещё не делал покупок -> начислить бонус пригласившему
        ref = user_tokens[uid].get("referrer")
        if ref and not user_tokens[uid].get("has_made_purchase", False):
            inviter = ref
            inviter_bonus = REFERRAL_BONUS_FUNC(tokens_to_add)
            if inviter_bonus > 0:
                add_stars(int(inviter), inviter_bonus)
            # пометить, что приглашённый совершил покупку — не давать повторно
            mark_made_purchase(user_id)

        # сообщение пользователю
        await message.answer(
            f"✅ Оплата принята. Вам зачислено {stars}⭐.\n"
            f"↪ Автоконвертация: {tokens_to_add} токенов (осталось {remainder_stars}⭐ на балансе).\n"
            f"💰 Баланс: {get_tokens(user_id)} токен(ов), {get_stars(user_id)}⭐"
        )

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

    # стоимость модели (в токенах)
    model = user_model[user_id]
    cost = MODEL_COSTS.get(model, 1)

    if get_tokens(user_id) < cost:
        await message.answer(f"❌ У вас недостаточно токенов. Для модели {model} требуется {cost} токен(ов). Купите звёзды и конвертируйте их в токены.", reply_markup=get_buy_stars_keyboard())
        return

    # списываем токены
    if not remove_tokens(user_id, cost):
        await message.answer("❌ Не удалось списать токены. Попробуйте ещё раз.")
        return

    # сохраняем в историю
    user_history[user_id].append(match_text)
    if len(user_history[user_id]) > 200:
        user_history[user_id] = user_history[user_id][-200:]

    await message.answer("🤖 Генерирую прогноз...")
    forecast = await ask_openai(match_text, model)

    await message.answer(
        f"📊 *Прогноз* (модель {model}, стоимость {cost} токенов):\n{forecast}\n\n"
        f"💰 Остаток токенов: {get_tokens(user_id)}",
        parse_mode="Markdown",
        reply_markup=get_feedback_buttons(match_text)
    )

# ==================== 👍👎 Фидбек и остальные коллбэки ====================
@dp.callback_query(F.data == "today_matches")
async def today_matches(callback: CallbackQuery):
    await callback.answer()
    matches = await fetch_matches_today()
    # сохраняем для пользователя
    user_last_matches[callback.from_user.id] = matches
    text = "📅 *Ближайшие матчи:*\n\n" + "\n".join(f"{i+1}. {m}" for i, m in enumerate(matches))
    # Добавляем кнопку сделать прогноз (переходим к выбору матча)
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
    # создаём кнопки списка матчей
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
    model = user_model[user_id]
    cost = MODEL_COSTS.get(model, 1)

    # Подтверждение: покажем стоимость и кнопку "Подтвердить"
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
    model = user_model[user_id]
    cost = MODEL_COSTS.get(model, 1)

    if get_tokens(user_id) < cost:
        await callback.answer("У вас недостаточно токенов для этой модели.", show_alert=True)
        return

    # списываем и генерируем
    if not remove_tokens(user_id, cost):
        await callback.answer("Не удалось списать токены.", show_alert=True)
        return

    await callback.message.answer("🤖 Генерирую прогноз...")
    forecast = await ask_openai(match_text, model)

    # лог истории
    user_history[user_id].append(f"{match_text} — модель {model}")
    if len(user_history[user_id]) > 200:
        user_history[user_id] = user_history[user_id][-200:]

    await callback.message.answer(
        f"📊 *Прогноз* (модель {model}):\n{forecast}\n\n"
        f"💰 Остаток токенов: {get_tokens(user_id)}",
        parse_mode="Markdown",
        reply_markup=get_feedback_buttons(match_text)
    )
    await callback.answer()

@dp.callback_query(F.data == "cancel")
async def cancel_cb(callback: CallbackQuery):
    await callback.answer("Отменено.", show_alert=False)

@dp.callback_query(F.data == "history")
async def history(callback: CallbackQuery):
    await callback.answer()
    history_list = user_history.get(callback.from_user.id, [])
    if not history_list:
        await callback.message.answer("🕒 У вас пока нет истории.")
    else:
        await callback.message.answer("🕒 *Последние прогнозы:*\n" + "\n".join(f"• {m}" for m in history_list[-10:]), parse_mode="Markdown")

@dp.callback_query(F.data == "feedback_report")
async def feedback_report(callback: CallbackQuery):
    await callback.answer()
    if not feedback_stats:
        await callback.message.answer("📊 Пока нет фидбека.")
    else:
        report = "\n".join(f"• {match} — 👍 {data['agree']} | 👎 {data['disagree']}" for match, data in feedback_stats.items())
        await callback.message.answer("📊 *Фидбек по прогнозам:*\n" + report, parse_mode="Markdown")

@dp.callback_query(F.data == "choose_model")
async def choose_model(callback: CallbackQuery):
    await callback.answer()
    await callback.message.answer("🧠 Выберите модель:", reply_markup=get_model_choice_keyboard())

@dp.callback_query(F.data.startswith("model:"))
async def set_model(callback: CallbackQuery):
    await callback.answer()
    model = callback.data.split(":", 1)[1]
    user_model[callback.from_user.id] = model
    await callback.message.answer(f"✅ Модель установлена: *{model}*", parse_mode="Markdown", reply_markup=get_main_menu(callback.from_user.id))

@dp.callback_query(F.data.startswith(("agree:", "disagree:")))
async def feedback_btn(callback: CallbackQuery):
    await callback.answer()
    action, match = callback.data.split(":", 1)
    if action in ("agree", "disagree"):
        feedback_stats[match][action] += 1
    reply = "👍 Спасибо за согласие!" if action == "agree" else "👎 Спасибо за честность!"
    await callback.message.answer(reply)

# Инструкция: Сделать прогноз (новая кнопка)
@dp.callback_query(F.data == "make_forecast")
async def make_forecast(callback: CallbackQuery):
    await callback.answer()
    await callback.message.answer(
        "📊 *Как сделать прогноз:*\n\n"
        "— Можно использовать кнопку *Ближайшие матчи* и выбрать конкретный матч.\n"
        "— Или отправить команду:\n"
        "/predict Команда1 - Команда2\n\n"
        f"⚠️ Стоимость прогноза зависит от выбранной модели. Текущая модель: {user_model[callback.from_user.id]}.\n"
        f"10⭐ = 1 токен. Пополнить баланс: нажмите «Пополнить баланс» в меню.",
        parse_mode="Markdown"
    )


@dp.callback_query(F.data == "how_it_works")
async def how_it_works(callback: CallbackQuery):
    await callback.answer()
    text = (
        "ℹ <b>Как это работает</b>\n\n"
        "1️⃣ При входе вы получаете 1 токен бесплатно.\n"
        "2️⃣ Подпишитесь на наш канал — получите ещё 1 токен.\n"
        "3️⃣ Токены тратите на прогнозы спортивных матчей.\n"
        "4️⃣ Чем дороже модель — тем точнее прогноз.\n\n"
        "💡 Пример:\n"
        "Вы выбираете матч <i>Барселона — Реал</i>.\n"
        "Бот анализирует статистику и даёт прогноз: победитель, возможный счёт, аргументы.\n"
        "Стоимость прогноза зависит от модели."
    )
    await callback.message.answer(text, parse_mode="HTML")


# ==================== ▶️ Запуск бота ====================
async def main():
    logging.info("✅ Бот запущен.")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
