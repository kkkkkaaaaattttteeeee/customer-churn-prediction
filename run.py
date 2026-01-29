"""
Основной скрипт для запуска проекта прогнозирования оттока клиентов
"""

import os
import sys


def print_project_structure():
    """Выводит структуру проекта"""
    print("=" * 60)
    print("СТРУКТУРА ПРОЕКТА")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Определяем что показывать
    show_items = [
        'data/',
        'notebooks/',
        'src/',
        'tests/',
        'configs/',
        '.gitignore',
        'README.md',
        'run.py'
    ]
    
    for item in show_items:
        item_path = os.path.join(project_root, item)
        if os.path.exists(item_path):
            if os.path.isdir(item_path):
                # Для папок показываем количество файлов
                file_count = len([f for f in os.listdir(item_path) 
                                if not f.startswith('.')])
                print(f"📁 {item} ({file_count} файлов)")
            else:
                print(f"📄 {item}")
        else:
            print(f"❌ {item} (отсутствует)")


def check_dependencies():
    """Проверяет наличие зависимостей"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    print("=" * 60)
    
    dependencies = [
        ('pandas', 'Для работы с данными'),
        ('numpy', 'Для численных операций'),
        ('sklearn', 'Для машинного обучения'),
        ('matplotlib', 'Для визуализации')
    ]
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✅ {package:15} - {description}")
        except ImportError:
            print(f"❌ {package:15} - {description} (не установлен)")


def show_next_steps():
    """Показывает следующие шаги для разработки"""
    print("\n" + "=" * 60)
    print("СЛЕДУЮЩИЕ ШАГИ")
    print("=" * 60)
    
    steps = [
        "1. Установить зависимости: pip install pandas numpy scikit-learn matplotlib",
        "2. Скачать датасет с Kaggle и положить в data/raw/",
        "3. Исследовать данные в notebooks/01_data_analysis.ipynb",
        "4. Реализовать feature engineering в src/features/",
        "5. Обучить модель в src/models/",
        "6. Протестировать код в tests/",
        "7. Создать API в api/"
    ]
    
    for step in steps:
        print(f"  {step}")


def test_data_module():
    """Тестирует модуль работы с данными"""
    print("\n" + "=" * 60)
    print("ТЕСТ МОДУЛЯ ДАННЫХ")
    print("=" * 60)
    
    try:
        # Добавляем src в путь
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from data import preprocessing
        
        print("✅ Модуль preprocessing загружен")
        
        # Пробуем загрузить тестовые данные
        test_path = 'data/raw/sample_data.csv'
        if os.path.exists(test_path):
            print(f"📊 Загружаем тестовые данные из {test_path}")
            data = preprocessing.load_data(test_path)
            
            if data and not isinstance(data, dict):
                print(f"   Загружено: {len(data)} строк")
                print(f"   Столбцы: {list(data.columns)}")
            else:
                print("   Данные не загружены (тестовый режим)")
        else:
            print(f"⚠️  Тестовые данные не найдены: {test_path}")
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании модуля: {e}")


def main():
    """Основная функция"""
    print("\n🚀 ПРОЕКТ: ПРОГНОЗИРОВАНИЕ ОТТОКА КЛИЕНТОВ")
    print("   Репозиторий: https://github.com/ваш-username/customer-churn-prediction")
    
    print_project_structure()
    check_dependencies()
    test_data_module()
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("✅ Проект готов к разработке!")
    print("=" * 60)


if __name__ == "__main__":
    main()