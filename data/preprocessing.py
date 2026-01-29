"""
Модуль для предобработки данных оттока клиентов
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """
    Загружает данные из CSV файла
    
    Parameters:
    -----------
    filepath : str
        Путь к CSV файлу с данными
        
    Returns:
    --------
    pd.DataFrame
        Загруженный DataFrame
    """
    print(f"Загрузка данных из {filepath}")
    df = pd.read_csv(filepath)
    print(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет пропущенные значения в DataFrame
    
    Returns:
    --------
    pd.DataFrame
        Таблица с информацией о пропусках
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_percent
    })
    
    return missing_df[missing_df['missing_count'] > 0]


def prepare_target(df: pd.DataFrame, target_column: str = 'Churn') -> pd.DataFrame:
    """
    Подготавливает целевую переменную (кодирует в бинарный формат)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame с преобразованной целевой переменной
    """
    df = df.copy()
    
    if target_column in df.columns:
        # Кодируем Yes/No в 1/0
        df[target_column] = df[target_column].map({'Yes': 1, 'No': 0})
        print(f"Целевая переменная '{target_column}' преобразована в бинарный формат")
    
    return df


def split_data(df: pd.DataFrame, target_column: str = 'Churn', 
               test_size: float = 0.2, random_state: int = 42):
    """
    Разделяет данные на обучающую и тестовую выборки
    
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    if target_column not in df.columns:
        raise ValueError(f"Целевая переменная '{target_column}' не найдена в данных")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Данные разделены:")
    print(f"  Обучающая выборка: {len(X_train)} строк")
    print(f"  Тестовая выборка: {len(X_test)} строк")
    print(f"  Распределение классов в train: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  Распределение классов в test: {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Модуль preprocessing загружен")