# Customer Churn Prediction

Проект для прогнозирования оттока клиентов в телекоммуникационной компании с использованием машинного обучения.

## Использование

Как пользователь, я хочу предсказывать вероятность ухода клиентов из компании, чтобы:
- Выявлять клиентов с высоким риском оттока
- Принимать проактивные меры по их удержанию
- Оптимизировать маркетинговые затраты на удержание

## Установка и настройка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/[ваш-username]/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Скачайте датасет [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) и поместите его в папку `data/raw/`

4. Для обучения модели запустите:
```bash
python src/models/train.py
```

5. Для запуска API сервиса:
```bash
uvicorn api.main:app --reload
```

После запуска API документация будет доступна по адресу: `http://localhost:8000/docs`

## Локальная разработка

Для разработки проекта:

1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```

2. Установите зависимости для разработки:
```bash
pip install -r requirements.txt
```

3. Запустите тесты:
```bash
pytest tests/
```

## Технические детали

- Датасет: 7043 клиента, 21 признак
- Целевая переменная: Churn (отток)
- Метрики: ROC-AUC, Precision, Recall, F1-score
- Модели: LightGBM, XGBoost, Random Forest
- API: FastAPI для инференса модели