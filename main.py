import os
import tempfile
import hashlib
import traceback
import logging
from PIL import Image
from telebot import TeleBot, types
import torch
import open_clip
from database import init_db, save_user_style, save_style_vector, get_user_styles, check_duplicate_image

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Стили с русскими названиями
fashion_styles = {
    "avangard": "авангард",
    "babiy": "бабий стиль",
    "bokho": "бохо",
    "business_casual": "бизнес-кэжуал",
    "casual": "кэжуал",
    "classic": "классика",
    "dendy": "денди",
    "feminine": "феминный",
    "ethnicity": "этнический",
    "drama": "драматический",
    "grunge": "гранж",
    "jokey": "жокей",
    "military": "милитари",
    "minimalism": "минимализм",
    "old_money": "олд мани",
    "preppy": "преппи",
    "quiet_luxury": "тихая роскошь",
    "retro": "ретро",
    "romance": "романтический",
    "safari": "сафари",
    "sailor": "морской стиль",
    "smart_casual": "смарт-кэжуал",
    "strange": "странный",
    "vintazh": "винтаж"
}

# Инициализация модели
try:
    logger.info("🔄 Загружаем модель...")
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    checkpoint_path = r"C:\Users\KAWAKI\Desktop\иишка\AI work\fashion_clip_finetuned_epoch1.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Файл модели не найден: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    logger.info("✅ Модель успешно загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {str(e)}")
    raise

TELEGRAM_TOKEN = ""
bot = TeleBot(TELEGRAM_TOKEN)

# Сессии
user_sessions = {}
active_requests = set()


def calculate_image_hash(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Ошибка расчета хеша: {str(e)}")
        return None


def create_style_keyboard():
    try:
        markup = types.InlineKeyboardMarkup(row_width=3)
        buttons = [
            types.InlineKeyboardButton(
                fashion_styles[style],
                callback_data=f"style_{style}"
            ) for style in fashion_styles
        ]
        markup.add(*buttons)
        return markup
    except Exception as e:
        logger.error(f"Ошибка создания клавиатуры: {str(e)}")
        return None


def cleanup_session(user_id):
    try:
        if user_id in user_sessions:
            session = user_sessions[user_id]
            if 'image_path' in session and os.path.exists(session['image_path']):
                os.remove(session['image_path'])
            del user_sessions[user_id]
        if user_id in active_requests:
            active_requests.remove(user_id)
    except Exception as e:
        logger.error(f"Ошибка очистки сессии: {str(e)}")


@bot.message_handler(commands=['start'])
def send_welcome(message):
    try:
        bot.reply_to(message, "👋 Привет! Отправь мне фото одежды, и я определю стиль.")
    except Exception as e:
        logger.error(f"Ошибка в start: {str(e)}")
        bot.reply_to(message, "❌ Произошла ошибка. Пожалуйста, попробуйте позже.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    user_id = message.from_user.id

    if user_id in active_requests:
        bot.reply_to(message, "⏳ Ваш предыдущий запрос еще обрабатывается...")
        return

    active_requests.add(user_id)
    temp_file_path = None

    try:
        # Скачивание фото
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Сохранение во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(downloaded_file)
            temp_file_path = tmp_file.name

        # Проверка хеша
        image_hash = calculate_image_hash(temp_file_path)
        if not image_hash:
            raise ValueError("Не удалось вычислить хеш изображения")

        # Проверка на дубликаты в БД
        matched_style = check_duplicate_image(user_id, image_hash)
        if matched_style:
            bot.send_message(
                message.chat.id,
                f"📌 Вы уже отправляли это фото. Стиль: *{fashion_styles.get(matched_style, matched_style)}*",
                parse_mode="Markdown"
            )
            return

        # Обработка изображения
        image = Image.open(temp_file_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_tokens = open_clip.tokenize(list(fashion_styles.keys()))
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T)
            probs = similarity.softmax(dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, 3)

        # Сохраняем топ-3 стиля с вероятностями
        top_styles = []
        for i in range(3):
            style = list(fashion_styles.keys())[top_indices[i]]
            prob = round(top_probs[i].item() * 100, 1)
            top_styles.append((style, prob))

        # Сохраняем сессию
        user_sessions[user_id] = {
            'features': image_features,
            'image_path': temp_file_path,
            'image_hash': image_hash,
            'message_id': message.message_id,
            'top_styles': top_styles
        }

        # Формируем сообщение с топ-3 стилями (стили выделены жирным)
        styles_message = "\n".join(
            [f"• *{fashion_styles[style]}*: {prob}%" for style, prob in top_styles]
        )

        # Создаем клавиатуру
        markup = types.InlineKeyboardMarkup()
        markup.row(
            types.InlineKeyboardButton("✅ Подходит", callback_data=f"accept_{top_styles[0][0]}"),
            types.InlineKeyboardButton("❌ Не подходит", callback_data="dislike")
        )

        # Отправляем результат
        sent_msg = bot.send_message(
            message.chat.id,
            f"🎨 Топ-3 предполагаемых стиля:\n{styles_message}\n\nПервый вариант вам подходит?",
            parse_mode="Markdown",
            reply_markup=markup
        )
        user_sessions[user_id]['bot_message_id'] = sent_msg.message_id

    except Exception as e:
        logger.error(f"Ошибка обработки фото: {str(e)}\n{traceback.format_exc()}")
        bot.reply_to(message, "❌ Произошла ошибка при обработке фото. Пожалуйста, попробуйте другое изображение.")
        if user_id in active_requests:
            active_requests.remove(user_id)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Ошибка удаления временного файла: {str(e)}")
        # Убедимся, что пользователь удален из active_requests, если обработка завершена
        if user_id in active_requests and (user_id not in user_sessions or 'bot_message_id' in user_sessions[user_id]):
            active_requests.remove(user_id)


@bot.callback_query_handler(func=lambda call: True)
def handle_all_callbacks(call):
    try:
        user_id = call.from_user.id
        logger.info(f"Получен callback: {call.data} от пользователя {user_id}")

        if user_id not in user_sessions:
            bot.answer_callback_query(call.id, "❌ Сессия устарела. Отправьте фото заново.", show_alert=True)
            return

        if call.data.startswith("accept_"):
            handle_accept_style(call)
        elif call.data == "dislike":
            handle_dislike_style(call)
        elif call.data.startswith("style_"):
            handle_select_style(call)
        else:
            bot.answer_callback_query(call.id, "❌ Неизвестная команда", show_alert=True)

    except Exception as e:
        logger.error(f"Ошибка обработки callback: {e}\n{traceback.format_exc()}")
        bot.answer_callback_query(call.id, "⚠️ Произошла ошибка. Попробуйте снова.", show_alert=True)
        if user_id in active_requests:
            active_requests.remove(user_id)


def handle_accept_style(call):
    user_id = call.from_user.id
    style_key = call.data[len("accept_"):]
    style_name = fashion_styles.get(style_key, style_key)

    if user_id not in user_sessions:
        bot.answer_callback_query(call.id, "❌ Сессия устарела", show_alert=True)
        return

    session = user_sessions[user_id]
    try:
        save_style_vector(
            user_id=user_id,
            style=style_key,
            vector=session['features'][0].cpu().numpy().tobytes(),
            image_hash=session['image_hash']
        )
        save_user_style(user_id, style_key)

        bot.answer_callback_query(call.id, f"✅ {style_name} сохранен!")
        bot.edit_message_text(
            f"✨ Отлично! Стиль *{style_name}* добавлен в вашу коллекцию.",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Ошибка сохранения стиля: {e}")
        raise
    finally:
        cleanup_session(user_id)


def handle_dislike_style(call):
    user_id = call.from_user.id

    if user_id not in user_sessions:
        bot.answer_callback_query(call.id, "❌ Сессия устарела", show_alert=True)
        return

    try:
        session = user_sessions[user_id]
        top_styles = session.get('top_styles', [])

        if not top_styles:
            raise ValueError("Не найдены стили для выбора")

        # Формируем сообщение с топ-3 стилями (стили выделены жирным)
        styles_message = "\n".join(
            [f"• *{fashion_styles[style]}*: {prob}%" for style, prob in top_styles]
        )

        markup = create_style_keyboard()
        if not markup:
            raise ValueError("Не удалось создать клавиатуру")

        bot.answer_callback_query(call.id)
        bot.edit_message_text(
            f"👎 Вы отказались от первого варианта.\n\nТоп-3 стиля:\n{styles_message}\n\nВыберите подходящий стиль:",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=markup,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Ошибка обработки отказа: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка загрузки стилей", show_alert=True)
        if user_id in active_requests:
            active_requests.remove(user_id)


def handle_select_style(call):
    user_id = call.from_user.id
    style_key = call.data[len("style_"):]
    style_name = fashion_styles.get(style_key, style_key)

    if user_id not in user_sessions:
        bot.answer_callback_query(call.id, "❌ Сессия устарела", show_alert=True)
        return

    try:
        session = user_sessions[user_id]
        save_style_vector(
            user_id=user_id,
            style=style_key,
            vector=session['features'][0].cpu().numpy().tobytes(),
            image_hash=session['image_hash']
        )
        save_user_style(user_id, style_key)

        bot.answer_callback_query(call.id, f"✅ Выбрано: {style_name}")
        bot.edit_message_text(
            f"💾 Стиль *{style_name}* успешно сохранён!",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Ошибка сохранения выбранного стиля: {e}")
        bot.answer_callback_query(call.id, "❌ Ошибка сохранения", show_alert=True)
    finally:
        cleanup_session(user_id)


if __name__ == '__main__':
    try:
        init_db()
        logger.info("🤖 Бот запущен и готов к работе!")
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Фатальная ошибка: {str(e)}\n{traceback.format_exc()}")
