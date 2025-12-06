import os
import numpy as np
from catboost import CatBoostRanker

class ModelHandler:
    """Обработчик для работы с CatBoost моделью"""
    
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Загружает CatBoost модель"""
        try:
            model_path = '/models/catboost_ranker.bin'
            print(f"🔄 Загрузка модели из {model_path}...")
            self.model = CatBoostRanker()
            self.model.load_model(model_path, format='cbm')
            print(f"✅ Модель загружена. Деревьев: {self.model.tree_count_}")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def prepare_features(self, user_id: int, item_ids: list):
        """Подготавливает признаки для модели"""
        # В реальном проекте здесь нужно брать признаки из данных
        # Сейчас создаем тестовые признаки
        features = []
        for item_id in item_ids:
            # Категориальные признаки (первые 6)
            cat_features = [
                user_id % 10,          # категория пользователя
                item_id % 10,          # категория товара
                (user_id + item_id) % 5,
                (user_id * 2) % 7,
                (item_id * 3) % 8,
                (user_id + item_id * 2) % 9
            ]
            
            # Числовые признаки (остальные 14)
            num_features = [
                user_id / 1000.0,
                item_id / 1000.0,
                np.sin(user_id),
                np.cos(item_id),
                (user_id % 100) / 100.0,
                (item_id % 100) / 100.0,
                np.log1p(user_id),
                np.log1p(item_id),
                user_id % 23 / 22.0,
                item_id % 17 / 16.0,
                (user_id * item_id) % 31 / 30.0,
                user_id % 3,
                item_id % 4,
                (user_id + item_id) % 5 / 4.0
            ]
            
            features.append(cat_features + num_features)
        
        return np.array(features, dtype=object)
    
    def predict(self, user_id: int, item_ids: list):
        """Делает предсказания"""
        if not self.model:
            raise ValueError("Модель не загружена")
        
        features = self.prepare_features(user_id, item_ids)
        predictions = self.model.predict(features)
        
        results = []
        for i, (item_id, score) in enumerate(zip(item_ids, predictions)):
            results.append({
                "item_id": int(item_id),
                "score": float(score),
                "user_id": int(user_id)
            })
        
        # Сортируем по убыванию score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 10):
        """Получает рекомендации для пользователя"""
        # Генерируем тестовые товары
        test_item_ids = list(range(100, 100 + n_recommendations * 2))
        
        predictions = self.predict(user_id, test_item_ids)
        return predictions[:n_recommendations]

# Создаем глобальный экземпляр
model_handler = ModelHandler()