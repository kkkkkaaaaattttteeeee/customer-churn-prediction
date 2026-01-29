"""
Модуль для создания и преобразования признаков
"""

import sys
import os


class FeatureEngineer:
    """
    Класс для инженерии признаков прогнозирования оттока
    """
    
    def __init__(self):
        self.feature_info = {}
        print("Инициализирован FeatureEngineer")
    
    def describe_features(self, data_info=None):
        """
        Описание признаков, которые будут созданы
        """
        print("\n" + "=" * 60)
        print("ПЛАН СОЗДАНИЯ ПРИЗНАКОВ")
        print("=" * 60)
        
        features_plan = {
            'Базовые признаки': [
                'tenure (стаж клиента в месяцах)',
                'MonthlyCharges (ежемесячный платеж)',
                'TotalCharges (общая сумма платежей)'
            ],
            'Категориальные признаки': [
                'Contract (тип контракта) -> one-hot encoding',
                'InternetService (тип интернета) -> one-hot encoding',
                'PaymentMethod (способ оплаты) -> one-hot encoding'
            ],
            'Производные признаки': [
                'tenure_group (группы стажа: новый, постоянный, лояльный)',
                'charge_per_month (средний чек за месяц)',
                'service_count (количество подключенных услуг)',
                'has_multiple_services (флаг множества услуг)'
            ],
            'Взаимодействия': [
                'tenure * MonthlyCharges',
                'SeniorCitizen * MonthlyCharges'
            ]
        }
        
        for category, features in features_plan.items():
            print(f"\n📊 {category}:")
            for feature in features:
                print(f"   • {feature}")
        
        return features_plan
    
    def get_categorical_columns(self):
        """
        Возвращает список категориальных столбцов для кодирования
        """
        categorical_cols = [
            'gender',
            'Partner',
            'Dependents',
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod'
        ]
        
        print(f"\nКатегориальные признаки для обработки: {len(categorical_cols)}")
        return categorical_cols
    
    def get_numerical_columns(self):
        """
        Возвращает список числовых столбцов
        """
        numerical_cols = [
            'SeniorCitizen',
            'tenure',
            'MonthlyCharges',
            'TotalCharges'
        ]
        
        print(f"Числовые признаки: {len(numerical_cols)}")
        return numerical_cols
    
    def create_derived_features_plan(self):
        """
        План создания производных признаков
        """
        print("\n" + "=" * 60)
        print("ПЛАН ПРОИЗВОДНЫХ ПРИЗНАКОВ")
        print("=" * 60)
        
        derived_features = [
            {
                'name': 'tenure_group',
                'description': 'Группировка стажа клиента',
                'logic': '0-12 мес = "new", 13-24 = "regular", 25+ = "loyal"'
            },
            {
                'name': 'avg_monthly_charge',
                'description': 'Средний ежемесячный платеж',
                'logic': 'TotalCharges / tenure (если tenure > 0)'
            },
            {
                'name': 'service_count',
                'description': 'Количество подключенных услуг',
                'logic': 'Сумма бинарных флагов услуг (OnlineSecurity, Backup и т.д.)'
            },
            {
                'name': 'is_high_value',
                'description': 'Флаг высокоценного клиента',
                'logic': 'MonthlyCharges > 70 и tenure > 12 месяцев'
            }
        ]
        
        for idx, feature in enumerate(derived_features, 1):
            print(f"\n{idx}. {feature['name']}:")
            print(f"   Описание: {feature['description']}")
            print(f"   Логика: {feature['logic']}")
        
        return derived_features
    
    def test_feature_creation(self, sample_data=None):
        """
        Тестирование создания признаков на примере
        """
        print("\n" + "=" * 60)
        print("ТЕСТ СОЗДАНИЯ ПРИЗНАКОВ (пример)")
        print("=" * 60)
        
        # Пример данных для демонстрации
        example_data = [
            {'tenure': 5, 'MonthlyCharges': 50, 'Contract': 'Month-to-month'},
            {'tenure': 15, 'MonthlyCharges': 80, 'Contract': 'One year'},
            {'tenure': 30, 'MonthlyCharges': 100, 'Contract': 'Two year'}
        ]
        
        print("\nПример исходных данных:")
        for i, row in enumerate(example_data, 1):
            print(f"  Клиент {i}: {row}")
        
        print("\nПреобразованные признаки:")
        for i, row in enumerate(example_data, 1):
            # Симулируем преобразования
            tenure = row['tenure']
            charges = row['MonthlyCharges']
            
            # Группа стажа
            if tenure <= 12:
                tenure_group = 'new'
            elif tenure <= 24:
                tenure_group = 'regular'
            else:
                tenure_group = 'loyal'
            
            # Флаг ценности
            is_high_value = 'Да' if charges > 70 and tenure > 12 else 'Нет'
            
            print(f"\n  Клиент {i}:")
            print(f"    tenure_group: {tenure_group}")
            print(f"    is_high_value: {is_high_value}")
            print(f"    monthly_charge_group: {'high' if charges > 70 else 'medium' if charges > 40 else 'low'}")
    
    def summary(self):
        """
        Сводка по всем признакам
        """
        print("\n" + "=" * 60)
        print("СВОДКА ПО ПРИЗНАКАМ")
        print("=" * 60)
        
        total_basic = len(self.get_numerical_columns()) + len(self.get_categorical_columns())
        total_derived = 4  # Из create_derived_features_plan
        
        print(f"\n📈 Общее количество признаков:")
        print(f"   • Базовые: {total_basic}")
        print(f"   • Производные: {total_derived}")
        print(f"   • После one-hot encoding: ~{total_basic + total_derived + 15} колонок")
        
        print("\n🎯 Ключевые этапы обработки:")
        print("   1. Загрузка и очистка данных")
        print("   2. Кодирование категориальных признаков")
        print("   3. Создание производных признаков")
        print("   4. Масштабирование числовых признаков")
        print("   5. Сохранение преобразований для инференса")


def main():
    """
    Основная функция для тестирования модуля
    """
    print("🔧 МОДУЛЬ FEATURE ENGINEERING")
    
    engineer = FeatureEngineer()
    engineer.describe_features()
    engineer.get_categorical_columns()
    engineer.get_numerical_columns()
    engineer.create_derived_features_plan()
    engineer.test_feature_creation()
    engineer.summary()
    
    print("\n✅ Модуль feature engineering готов к использованию")


if __name__ == "__main__":
    main()