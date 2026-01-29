"""
Главный скрипт проекта: прогнозирование оттока клиентов
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

def main():
    print("=" * 70)
    print("🚀 ПРОГНОЗИРОВАНИЕ ОТТОКА КЛИЕНТОВ - ML ПРОЕКТ")
    print("=" * 70)
    
    # 1. Загрузка данных
    print("\n1. 📂 ЗАГРУЗКА ДАННЫХ")
    data_path = "data/raw/telco_churn.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Файл не найден: {data_path}")
        print("📥 Скачайте датасет с Kaggle:")
        print("   https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Загружено: {df.shape[0]} клиентов, {df.shape[1]} признаков")
    
    # 2. Предобработка
    print("\n2. 🔧 ПРЕДОБРАБОТКА ДАННЫХ")
    
    # Удаляем ID
    df = df.drop('customerID', axis=1)
    
    # Исправляем TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Создаем простой признак: количество услуг
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies']
    
    def count_services(row):
        count = 0
        for col in service_cols:
            if col in df.columns:
                if str(row[col]) not in ['No', 'No internet service']:
                    count += 1
        return count
    
    df['service_count'] = df.apply(count_services, axis=1)
    
    # Кодируем категориальные признаки
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'Churn':  # Целевую переменную обработаем отдельно
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Кодируем целевую переменную
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print(f"✅ Создан новый признак: service_count")
    print(f"✅ Закодировано категориальных признаков: {len(categorical_cols)-1}")
    
    # 3. Подготовка данных для ML
    print("\n3. 🎯 ПОДГОТОВКА ДЛЯ ML")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Масштабируем признаки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Обучающая выборка: {X_train.shape[0]} клиентов")
    print(f"✅ Тестовая выборка: {X_test.shape[0]} клиентов")
    print(f"✅ Отток в тестовой выборке: {y_test.mean()*100:.1f}%")
    
    # 4. Обучение моделей
    print("\n4. 🧠 ОБУЧЕНИЕ ML МОДЕЛЕЙ")
    
    # Логистическая регрессия
    print("\n   📊 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    print(f"      Accuracy:  {accuracy_score(y_test, y_pred_lr):.3f}")
    print(f"      Precision: {precision_score(y_test, y_pred_lr):.3f}")
    print(f"      Recall:    {recall_score(y_test, y_pred_lr):.3f}")
    print(f"      F1-Score:  {f1_score(y_test, y_pred_lr):.3f}")
    
    # Случайный лес
    print("\n   🌲 СЛУЧАЙНЫЙ ЛЕС")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    
    print(f"      Accuracy:  {accuracy_score(y_test, y_pred_rf):.3f}")
    print(f"      Precision: {precision_score(y_test, y_pred_rf):.3f}")
    print(f"      Recall:    {recall_score(y_test, y_pred_rf):.3f}")
    print(f"      F1-Score:  {f1_score(y_test, y_pred_rf):.3f}")
    
    # 5. Сохранение моделей
    print("\n5. 💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    
    os.makedirs('models', exist_ok=True)
    
    with open('models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Модели сохранены в папке 'models/'")
    
    # 6. Создание отчета
    print("\n6. 📊 СОЗДАНИЕ ОТЧЕТА")
    
    os.makedirs('reports', exist_ok=True)
    
    report = f"""
# 📊 ОТЧЕТ ML ПРОЕКТА

## 🎯 Прогнозирование оттока клиентов

### 📈 Результаты

**Логистическая регрессия:**
- Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}
- Precision: {precision_score(y_test, y_pred_lr):.3f}
- Recall: {recall_score(y_test, y_pred_lr):.3f}
- F1-Score: {f1_score(y_test, y_pred_lr):.3f}

**Случайный лес:**
- Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}
- Precision: {precision_score(y_test, y_pred_rf):.3f}
- Recall: {recall_score(y_test, y_pred_rf):.3f}
- F1-Score: {f1_score(y_test, y_pred_rf):.3f}

### 📁 Файлы проекта
- Модели: `models/lr_model.pkl`, `models/rf_model.pkl`
- Scaler: `models/scaler.pkl`
- Данные: `data/raw/telco_churn.csv`
- Код: `main.py`

### 🚀 Как использовать
1. Установите зависимости: `pip install pandas scikit-learn`
2. Запустите: `python main.py`
3. Модели будут обучены и сохранены

---
*Проект создан для демонстрации навыков ML-инженера*
"""
    
    with open('reports/project_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Отчет сохранен: 'reports/project_report.md'")
    
    # 7. Итог
    print("\n" + "=" * 70)
    print("🎉 ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
    print("=" * 70)
    
    print(f"\n📊 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   • Обработано данных: {df.shape[0]} клиентов")
    print(f"   • Обучено моделей: 2")
    print(f"   • Лучшая F1-Score: {max(f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)):.3f}")
    
    print(f"\n🎯 ЧТО ПОКАЗЫВАЕТ ПРОЕКТ:")
    print(f"   • Умение работать с реальными данными")
    print(f"   • Навыки предобработки и feature engineering")
    print(f"   • Опыт обучения и оценки ML моделей")
    print(f"   • Умение сохранять и документировать результаты")
    
    print(f"\n📁 СТРУКТУРА ПРОЕКТА:")
    print(f"   customer-churn-prediction/")
    print(f"   ├── data/raw/telco_churn.csv")
    print(f"   ├── notebooks/01_data_analysis.ipynb")
    print(f"   ├── models/                    # Обученные модели")
    print(f"   ├── reports/                   # Отчеты")
    print(f"   ├── main.py                    # Главный скрипт")
    print(f"   ├── README.md                  # Документация")
    print(f"   └── requirements.txt           # Зависимости")

if __name__ == "__main__":
    main()