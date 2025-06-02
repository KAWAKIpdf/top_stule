import sqlite3
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def init_db():
    try:
        with sqlite3.connect('fashion_styles.db') as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_styles (
                    user_id INTEGER NOT NULL,
                    style TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, style)
                )
            ''')


            cursor.execute('''
                CREATE TABLE IF NOT EXISTS style_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    style TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    image_hash TEXT NOT NULL UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id, style) REFERENCES user_styles(user_id, style)
                )
            ''')

            conn.commit()
        logger.info("✅ База данных успешно инициализирована")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации БД: {str(e)}")
        raise

def save_user_style(user_id: int, style: str) -> None:
    try:
        with sqlite3.connect('fashion_styles.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR IGNORE INTO user_styles (user_id, style) VALUES (?, ?)',
                (user_id, style))
            conn.commit()
            logger.info(f"💾 Стиль {style} сохранен для пользователя {user_id}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения стиля пользователя: {str(e)}")
        raise

def save_style_vector(user_id: int, style: str, vector: bytes, image_hash: str) -> None:
    try:
        with sqlite3.connect('fashion_styles.db') as conn:
            cursor = conn.cursor()

            save_user_style(user_id, style)

            cursor.execute(
                '''INSERT INTO style_vectors 
                (user_id, style, vector, image_hash) 
                VALUES (?, ?, ?, ?)''',
                (user_id, style, vector, image_hash))
            conn.commit()
        logger.info(f"📊 Вектор стиля сохранен для пользователя {user_id}")
    except sqlite3.IntegrityError:
        logger.warning(f"⚠️ Изображение с хэшем {image_hash} уже существует в БД")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения вектора стиля: {str(e)}")
        raise

def get_user_styles(user_id: int) -> List[Dict]:
    try:
        with sqlite3.connect('fashion_styles.db') as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT style, timestamp 
                FROM user_styles 
                WHERE user_id = ? 
                ORDER BY timestamp DESC''',
                (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"❌ Ошибка получения стилей пользователя: {str(e)}")
        return []

def check_duplicate_image(user_id: int, image_hash: str) -> Optional[str]:
    try:
        with sqlite3.connect('fashion_styles.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT style 
                FROM style_vectors 
                WHERE image_hash = ? AND user_id = ?''',
                (image_hash, user_id))
            result = cursor.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"❌ Ошибка проверки дубликата: {str(e)}")
        return None



def get_user_recent_style(user_id: int) -> Optional[str]:
    """Получение последнего стиля пользователя"""
    try:
        with sqlite3.connect('fashion_styles.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT style 
                FROM user_styles 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1''',
                (user_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"❌ Ошибка получения последнего стиля: {str(e)}")
        return None


def get_style_statistics() -> Dict[str, int]:
    """Статистика популярности стилей"""
    try:
        with sqlite3.connect('fashion_styles.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT style, COUNT(*) as count 
                FROM user_styles 
                GROUP BY style 
                ORDER BY count DESC''')
            return {row[0]: row[1] for row in cursor.fetchall()}
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {str(e)}")
        return {}
