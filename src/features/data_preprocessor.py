"""
Модуль для предобработки данных и создания признаков
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Класс для предобработки данных и создания признаков
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        print("Инициализирован DataPreprocessor")
    
    def load_data(self, filepath):
        """Загружает данные"""
        print(f"📂 Загрузка данных из: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов")
        return df
    
    def clean_data(self, df):
        """Очистка данных"""
        print("\n🧹 ОЧИСТКА ДАННЫХ:")
        print("=" * 50)
        
        df_clean = df.copy()
        
        # 1. Удаляем customerID - не информативный признак
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
            print("✅ Удален столбец: customerID")
        
        # 2. Исправляем TotalCharges (может быть строкой с пробелами)
        if 'TotalCharges' in df_clean.columns:
            # Преобразуем в числовой тип, заменяем пробелы на NaN
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            
            # Заменяем NaN медианой
            median_value = df_clean['TotalCharges'].median()
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_value)
            print(f"✅ Исправлен TotalCharges (NaN заполнены медианой: {median_value:.2f})")
        
        # 3. Преобразуем SeniorCitizen в строку для единообразия
        if 'SeniorCitizen' in df_clean.columns:
            df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
            print("✅ Преобразован SeniorCitizen в категориальный")
        
        print(f"📊 Размер после очистки: {df_clean.shape}")
        return df_clean
    
    def create_new_features(self, df):
        """Создание новых признаков"""
        print("\n🔧 СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ:")
        print("=" * 50)
        
        df_features = df.copy()
        
        # 1. Группы стажа (tenure groups)
        df_features['tenure_group'] = pd.cut(
            df_features['tenure'],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61+']
        )
        print("✅ Создан tenure_group (группы стажа)")
        
        # 2. Средний ежемесячный платеж
        df_features['avg_monthly_charge'] = np.where(
            df_features['tenure'] > 0,
            df_features['TotalCharges'] / df_features['tenure'],
            df_features['MonthlyCharges']
        )
        print("✅ Создан avg_monthly_charge (средний чек)")
        
        # 3. Количество услуг
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        available_services = [col for col in service_columns if col in df_features.columns]
        
        def count_services(row):
            count = 0
            for service in available_services:
                if row[service] not in ['No', 'No internet service', 'No phone service']:
                    count += 1
            return count
        
        df_features['service_count'] = df_features.apply(count_services, axis=1)
        print(f"✅ Создан service_count (количество услуг, использованы: {len(available_services)} признаков)")
        
        # 4. Флаги важных характеристик
        df_features['has_internet'] = df_features['InternetService'].apply(
            lambda x: 0 if x == 'No' else 1
        )
        print("✅ Создан has_internet (флаг наличия интернета)")
        
        df_features['has_streaming'] = df_features.apply(
            lambda row: 1 if row['StreamingTV'] == 'Yes' or row['StreamingMovies'] == 'Yes' else 0,
            axis=1
        )
        print("✅ Создан has_streaming (флаг стриминговых услуг)")
        
        # 5. Тип контракта (бинарный)
        contract_mapping = {
            'Month-to-month': 0,
            'One year': 1,
            'Two year': 2
        }
        df_features['contract_type'] = df_features['Contract'].map(contract_mapping)
        print("✅ Создан contract_type (кодированный тип контракта)")
        
        print(f"📊 Всего создано: 6 новых признаков")
        print(f"📊 Общее количество признаков: {df_features.shape[1]}")
        
        return df_features
    
    def encode_categorical_features(self, df):
        """Кодирование категориальных признаков"""
        print("\n🔠 КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:")
        print("=" * 50)
        
        df_encoded = df.copy()
        
        # Столбцы для one-hot encoding (бинарные или несколько категорий)
        one_hot_columns = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'PaperlessBilling', 'Churn'
        ]
        
        # Столбцы для label encoding (порядковые)
        label_encode_columns = [
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaymentMethod', 'tenure_group'
        ]
        
        # 1. One-hot encoding для бинарных признаков
        print("One-hot encoding для:")
        for col in one_hot_columns:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
                print(f"  ✅ {col}")
        
        # 2. Label encoding для признаков с несколькими категориями
        print("\nLabel encoding для:")
        for col in label_encode_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"  ✅ {col} ({len(le.classes_)} категорий)")
        
        print(f"\n📊 После кодирования: {df_encoded.shape[1]} столбцов")
        return df_encoded
    
    def scale_numerical_features(self, df):
        """Масштабирование числовых признаков"""
        print("\n📏 МАСШТАБИРОВАНИЕ ЧИСЛОВЫХ ПРИЗНАКОВ:")
        print("=" * 50)
        
        df_scaled = df.copy()
        
        # Числовые столбцы для масштабирования
        numerical_columns = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'avg_monthly_charge', 'service_count', 'contract_type'
        ]
        
        # Только существующие столбцы
        existing_numerical = [col for col in numerical_columns if col in df_scaled.columns]
        
        if existing_numerical:
            print(f"Масштабирование {len(existing_numerical)} признаков:")
            for col in existing_numerical:
                print(f"  🔧 {col}")
            
            # Сохраняем исходные значения
            original_values = df_scaled[existing_numerical].copy()
            
            # Масштабируем
            df_scaled[existing_numerical] = self.scaler.fit_transform(df_scaled[existing_numerical])
            
            print(f"✅ Признаки масштабированы (StandardScaler)")
            self.feature_columns = existing_numerical
        else:
            print("⚠️  Нет числовых признаков для масштабирования")
        
        return df_scaled
    
    def prepare_final_dataset(self, df, target_column='Churn'):
        """Подготовка финального датасета"""
        print("\n🎯 ПОДГОТОВКА ФИНАЛЬНОГО ДАТАСЕТА:")
        print("=" * 50)
        
        # Отделяем целевую переменную
        if target_column in df.columns:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            print(f"✅ Целевая переменная: {target_column}")
            print(f"📊 Признаки: {X.shape[1]} столбцов")
            print(f"📊 Целевая: {len(y)} значений")
            print(f"📊 Распределение классов:")
            print(f"   No (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
            print(f"   Yes (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
            
            return X, y
        else:
            print(f"❌ Целевая переменная '{target_column}' не найдена")
            return df, None
    
    def save_processed_data(self, X, y, prefix='churn'):
        """Сохранение обработанных данных"""
        import os
        
        # Создаем папку если нет
        os.makedirs('data/processed', exist_ok=True)
        
        # Сохраняем признаки и целевую переменную
        X_path = f'data/processed/{prefix}_features.csv'
        y_path = f'data/processed/{prefix}_target.csv'
        
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)
        
        print(f"\n💾 СОХРАНЕНИЕ ДАННЫХ:")
        print(f"  ✅ Признаки: {X_path} ({X.shape[0]} строк, {X.shape[1]} столбцов)")
        print(f"  ✅ Целевая: {y_path} ({len(y)} значений)")
        
        return X_path, y_path
    
    def full_pipeline(self, input_path, output_prefix='churn'):
        """Полный пайплайн предобработки"""
        print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ПРЕДОБРАБОТКИ")
        print("=" * 70)
        
        # 1. Загрузка
        df = self.load_data(input_path)
        
        # 2. Очистка
        df_clean = self.clean_data(df)
        
        # 3. Создание признаков
        df_features = self.create_new_features(df_clean)
        
        # 4. Кодирование
        df_encoded = self.encode_categorical_features(df_features)
        
        # 5. Масштабирование
        df_scaled = self.scale_numerical_features(df_encoded)
        
        # 6. Подготовка финального датасета
        X, y = self.prepare_final_dataset(df_scaled)
        
        # 7. Сохранение
        if y is not None:
            X_path, y_path = self.save_processed_data(X, y, output_prefix)
            
            print("\n" + "=" * 70)
            print("✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
            print("=" * 70)
            
            print(f"\n📈 ИТОГОВАЯ СТАТИСТИКА:")
            print(f"  Исходный датасет: {df.shape[1]} признаков")
            print(f"  Финальный датасет: {X.shape[1]} признаков")
            print(f"  Создано новых признаков: 6")
            print(f"  Закодировано категориальных признаков: ~15")
            
            return X, y, X_path, y_path
        else:
            return None


def main():
    """Основная функция для тестирования"""
    print("🧪 ТЕСТИРОВАНИЕ ПАЙПЛАЙНА ПРЕДОБРАБОТКИ")
    print("=" * 70)
    
    # Создаем препроцессор
    preprocessor = DataPreprocessor()
    
    # Запускаем пайплайн на тестовых данных
    input_path = "data/raw/sample_data.csv"  # Можно заменить на полный датасет
    
    print(f"\n📁 Входной файл: {input_path}")
    
    try:
        result = preprocessor.full_pipeline(input_path, 'test')
        
        if result:
            X, y, X_path, y_path = result
            print(f"\n✅ Тестирование завершено успешно!")
            print(f"   Признаки: {X.shape}")
            print(f"   Целевая: {y.shape}")
        else:
            print("❌ Тестирование не удалось")
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()