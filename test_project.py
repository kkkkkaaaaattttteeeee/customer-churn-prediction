"""
Тестовый скрипт для проверки структуры проекта
"""

import os


def check_project_structure():
    """Проверяет структуру проекта"""
    print("🔍 ПРОВЕРКА СТРУКТУРЫ ПРОЕКТА")
    print("=" * 50)
    
    # Ожидаемая структура
    expected_structure = {
        'data/': 'Папка с данными',
        'data/raw/': 'Исходные данные',
        'data/processed/': 'Обработанные данные',
        'notebooks/': 'Jupyter ноутбуки',
        'src/': 'Исходный код',
        'src/data/': 'Код для обработки данных',
        'src/features/': 'Код для feature engineering',
        'src/models/': 'Код для моделей ML',
        'src/visualization/': 'Код для визуализации',
        'tests/': 'Тесты',
        'configs/': 'Конфигурационные файлы'
    }
    
    # Файлы, которые должны быть
    expected_files = {
        'README.md': 'Документация проекта',
        '.gitignore': 'Игнорируемые файлы',
        'run.py': 'Основной скрипт запуска',
        'test_project.py': 'Этот тестовый скрипт'
    }
    
    # Проверяем папки
    print("\n📁 ПРОВЕРКА ПАПОК:")
    all_good = True
    
    for folder, description in expected_structure.items():
        if os.path.exists(folder):
            # Считаем файлы в папке (кроме скрытых и __pycache__)
            if os.path.isdir(folder):
                files_count = len([f for f in os.listdir(folder) 
                                  if not f.startswith('.') and f != '__pycache__'])
                print(f"  ✅ {folder:25} - {description} ({files_count} файлов)")
            else:
                print(f"  ❌ {folder:25} - не папка, а файл!")
                all_good = False
        else:
            print(f"  ❌ {folder:25} - отсутствует")
            all_good = False
    
    # Проверяем файлы
    print("\n📄 ПРОВЕРКА ФАЙЛОВ:")
    for file, description in expected_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✅ {file:25} - {description} ({size} байт)")
        else:
            print(f"  ❌ {file:25} - отсутствует")
            all_good = False
    
    # Проверяем файлы в src
    print("\n🔧 ПРОВЕРКА ИСХОДНОГО КОДА (src/):")
    src_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/preprocessing.py',
        'src/features/__init__.py',
        'src/features/feature_engineering.py',
        'src/models/__init__.py',
        'src/models/model_pipeline.py',
        'src/visualization/__init__.py'
    ]
    
    for file in src_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            all_good = False
    
    # Проверяем ноутбуки
    print("\n📓 ПРОВЕРКА НОУТБУКОВ:")
    notebook_files = [
        'notebooks/01_data_analysis.ipynb',
        'notebooks/02_test_data.ipynb'
    ]
    
    for nb in notebook_files:
        if os.path.exists(nb):
            print(f"  ✅ {nb}")
        else:
            print(f"  ⚠️  {nb} - отсутствует (можно создать позже)")
    
    return all_good


def check_python_environment():
    """Проверяет Python окружение"""
    print("\n🐍 ПРОВЕРКА PYTHON ОКРУЖЕНИЯ:")
    print("=" * 50)
    
    import sys
    
    print(f"  Версия Python: {sys.version}")
    print(f"  Путь к Python: {sys.executable}")
    print(f"  Кодировка: {sys.getdefaultencoding()}")
    
    # Проверяем текущую директорию
    print(f"  Текущая директория: {os.getcwd()}")
    
    # Проверяем структуру sys.path
    print(f"  Python path содержит проект: {os.getcwd() in sys.path}")


def show_next_actions():
    """Показывает следующие действия"""
    print("\n🚀 СЛЕДУЮЩИЕ ДЕЙСТВИЯ:")
    print("=" * 50)
    
    actions = [
        "1. Установить зависимости: pip install pandas numpy scikit-learn matplotlib seaborn",
        "2. Запустить тестовый скрипт: python test_project.py",
        "3. Проверить модули: python -c 'from src.data import preprocessing'",
        "4. Запустить основной скрипт: python run.py",
        "5. Открыть ноутбук для анализа: notebooks/01_data_analysis.ipynb",
        "6. Скачать полный датасет с Kaggle",
        "7. Начать реализацию ML пайплайна"
    ]
    
    for action in actions:
        print(f"  {action}")


def main():
    """Основная функция"""
    print("\n" + "=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ СТРУКТУРЫ ПРОЕКТА")
    print("=" * 60)
    
    # Проверяем структуру
    structure_ok = check_project_structure()
    
    # Проверяем окружение
    check_python_environment()
    
    # Показываем следующий шаги
    show_next_actions()
    
    # Итог
    print("\n" + "=" * 60)
    if structure_ok:
        print("✅ СТРУКТУРА ПРОЕКТА В ПОРЯДКЕ!")
        print("   Проект готов к разработке.")
    else:
        print("⚠️  ЕСТЬ ПРОБЛЕМЫ СО СТРУКТУРОЙ!")
        print("   Проверьте отсутствующие файлы/папки.")
    print("=" * 60)
    
    return structure_ok


if __name__ == "__main__":
    main()