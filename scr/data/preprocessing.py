"""
Модуль предобработки данных
"""

import pandas as pd
import numpy as np

def load_data(filepath):
    """Загрузка данных"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Очистка данных"""
    df_clean = df.copy()
    
    # Удаляем ID
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
    
    # Исправляем TotalCharges
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())
    
    return df_clean

def encode_target(df, target_col='Churn'):
    """Кодирование целевой переменной"""
    df_encoded = df.copy()
    df_encoded[target_col] = df_encoded[target_col].map({'Yes': 1, 'No': 0})
    return df_encoded