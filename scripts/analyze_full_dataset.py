"""
Полный анализ датасета Telco Customer Churn
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_and_analyze():
    """Загружает и анализирует датасет"""
    print("🔍 ПОЛНЫЙ АНАЛИЗ ДАТАСЕТА TELCO CUSTOMER CHURN")
    print("=" * 70)
    
    # Загружаем данные
    data_path = "data/raw/telco_churn.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Файл не найден: {data_path}")
        return None
    
    print(f"📂 Загрузка данных из: {data_path}")
    df = pd.read_csv(data_path)
    
    # Сохраняем копию для анализа
    df.to_csv("data/processed/churn_data_full.csv", index=False)
    
    return df

def basic_statistics(df):
    """Базовая статистика"""
    print("\n📊 БАЗОВАЯ СТАТИСТИКА:")
    print("=" * 70)
    
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    print(f"\n📋 Столбцы ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2}. {col}")
    
    print(f"\n📈 Типы данных:")
    print(df.dtypes)
    
    print(f"\n🔍 Пропущенные значения:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ Нет пропущенных значений")
    else:
        print("Пропущенные значения:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")

def analyze_target(df):
    """Анализ целевой переменной"""
    print("\n🎯 АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (Churn):")
    print("=" * 70)
    
    if 'Churn' not in df.columns:
        print("❌ Столбец 'Churn' не найден")
        return
    
    churn_counts = df['Churn'].value_counts()
    churn_percent = df['Churn'].value_counts(normalize=True) * 100
    
    print(f"\nРаспределение:")
    print(f"  No (без оттока):  {churn_counts['No']:>6} ({churn_percent['No']:5.1f}%)")
    print(f"  Yes (с оттоком):  {churn_counts['Yes']:>6} ({churn_percent['Yes']:5.1f}%)")
    
    print(f"\n📊 Дисбаланс классов: {churn_percent['Yes']:.1f}% vs {churn_percent['No']:.1f}%")
    print("   -> Нужно использовать метрики: Precision, Recall, F1, ROC-AUC")

def analyze_numeric_features(df):
    """Анализ числовых признаков"""
    print("\n🔢 АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ:")
    print("=" * 70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("❌ Числовые признаки не найдены")
        return
    
    print(f"Найдено числовых признаков: {len(numeric_cols)}")
    
    for col in numeric_cols:
        print(f"\n📊 {col}:")
        print(f"  Min: {df[col].min():.1f}, Max: {df[col].max():.1f}")
        print(f"  Mean: {df[col].mean():.1f}, Median: {df[col].median():.1f}")
        print(f"  Std: {df[col].std():.1f}")
        
        # Проверяем распределение
        if df[col].nunique() < 10:
            print(f"  Уникальных значений: {df[col].nunique()}")
            print(f"  Распределение: {df[col].value_counts().to_dict()}")

def analyze_categorical_features(df):
    """Анализ категориальных признаков"""
    print("\n🏷️ АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:")
    print("=" * 70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("❌ Категориальные признаки не найдены")
        return
    
    print(f"Найдено категориальных признаков: {len(categorical_cols)}")
    
    for col in categorical_cols:
        if col != 'customerID':  # Пропускаем ID
            unique_count = df[col].nunique()
            print(f"\n📋 {col}:")
            print(f"  Уникальных значений: {unique_count}")
            
            if unique_count <= 10:
                # Показываем распределение для признаков с малым количеством значений
                value_counts = df[col].value_counts()
                for value, count in value_counts.items():
                    percent = (count / len(df)) * 100
                    print(f"  {value}: {count:>5} ({percent:5.1f}%)")

def save_analysis_report(df, report_path="data/processed/analysis_report.txt"):
    """Сохраняет отчет анализа"""
    print(f"\n💾 СОХРАНЕНИЕ ОТЧЕТА...")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ АНАЛИЗА ДАННЫХ: TELCO CUSTOMER CHURN\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Общая информация:\n")
        f.write(f"- Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов\n")
        f.write(f"- Пропущенные значения: {df.isnull().sum().sum()}\n\n")
        
        f.write(f"Целевая переменная (Churn):\n")
        churn_counts = df['Churn'].value_counts()
        churn_percent = df['Churn'].value_counts(normalize=True) * 100
        f.write(f"- No: {churn_counts['No']} ({churn_percent['No']:.1f}%)\n")
        f.write(f"- Yes: {churn_counts['Yes']} ({churn_percent['Yes']:.1f}%)\n\n")
        
        f.write("Рекомендации для feature engineering:\n")
        f.write("1. Преобразовать категориальные признаки в числовые (one-hot encoding)\n")
        f.write("2. Обработать TotalCharges (сейчас строка, содержит пробелы)\n")
        f.write("3. Создать новые признаки: tenure groups, service count\n")
        f.write("4. Масштабировать числовые признаки\n")
        f.write("5. Учесть дисбаланс классов при обучении модели\n")
    
    print(f"✅ Отчет сохранен: {report_path}")

def main():
    """Основная функция"""
    print("🚀 ЗАПУСК АНАЛИЗА ДАННЫХ")
    print("=" * 70)
    
    # Загружаем данные
    df = load_and_analyze()
    
    if df is None:
        return
    
    # Выполняем анализ
    basic_statistics(df)
    analyze_target(df)
    analyze_numeric_features(df)
    analyze_categorical_features(df)
    
    # Сохраняем отчет
    save_analysis_report(df)
    
    print("\n" + "=" * 70)
    print("✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 70)
    
    print("\n📝 КЛЮЧЕВЫЕ ВЫВОДЫ:")
    print("1. Данные готовы для feature engineering")
    print("2. Есть дисбаланс классов (26.5% churn)")
    print("3. Требуется обработка категориальных признаков")
    print("4. Можно приступать к построению ML модели")

if __name__ == "__main__":
    main()