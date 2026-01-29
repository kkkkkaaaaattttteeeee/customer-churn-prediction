"""
Модуль для предобработки данных оттока клиентов
"""

import os
import sys

# Проверяем наличие библиотек
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    print("Библиотеки pandas/numpy не установлены. Используем тестовый режим.")
    HAS_PANDAS = False

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    print("Библиотека scikit-learn не установлена. Используем тестовый режим.")
    HAS_SKLEARN = False


def load_data(filepath: str):
    """
    Загружает данные из CSV файла
    """
    print(f"Загрузка данных из {filepath}")
    
    if not HAS_PANDAS:
        print("РЕЖИМ ТЕСТИРОВАНИЯ: pandas не установлен")
        # Возвращаем заглушку
        return {
            'status': 'test_mode',
            'filepath': filepath,
            'message': 'pandas не установлен, данные не загружены'
        }
    
    if not os.path.exists(filepath):
        print(f"Файл {filepath} не найден")
        return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"Успешно загружено: {len(df)} строк, {len(df.columns)} столбцов")
        print(f"Столбцы: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return None


def analyze_data(df):
    """
    Базовый анализ данных
    """
    if not HAS_PANDAS or df is None or isinstance(df, dict):
        print("Анализ данных недоступен в тестовом режиме")
        return
    
    print("\n=== АНАЛИЗ ДАННЫХ ===")
    print(f"Размер данных: {df.shape}")
    print(f"\nТипы данных:")
    print(df.dtypes)
    
    print(f"\nПервые 3 строки:")
    print(df.head(3))
    
    if 'Churn' in df.columns:
        churn_dist = df['Churn'].value_counts()
        print(f"\nРаспределение целевой переменной (Churn):")
        print(churn_dist)


def create_test_data():
    """
    Создает тестовые данные для отладки
    """
    print("Создание тестовых данных...")
    
    test_data = [
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,Churn",
        "001,Male,0,Yes,No,12,Yes,No",
        "002,Female,1,No,Yes,24,Yes,Yes",
        "003,Male,0,Yes,No,6,No,No"
    ]
    
    with open('data/raw/test_churn.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_data))
    
    print("Тестовые данные сохранены в data/raw/test_churn.csv")
    return 'data/raw/test_churn.csv'


def main():
    """
    Основная функция для тестирования модуля
    """
    print("=== МОДУЛЬ ПРЕДОБРАБОТКИ ДАННЫХ ===")
    
    # Проверяем наличие данных
    sample_path = 'data/raw/sample_data.csv'
    if os.path.exists(sample_path):
        print(f"Найден файл: {sample_path}")
        df = load_data(sample_path)
        if df is not None and not isinstance(df, dict):
            analyze_data(df)
    else:
        print(f"Файл {sample_path} не найден")
        # Создаем тестовые данные
        test_file = create_test_data()
        df = load_data(test_file)
        if df is not None and not isinstance(df, dict):
            analyze_data(df)


if __name__ == "__main__":
    main()