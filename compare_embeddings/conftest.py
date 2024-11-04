# conftest.py
def pytest_addoption(parser):
    parser.addoption("--show_data", action="store_true", help="enable printing of expected data in tests")
