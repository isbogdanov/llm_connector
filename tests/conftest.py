import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-local", action="store_true", default=False, help="run tests requiring local LLM models"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "local_model: mark test as requiring a local LLM instance")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-local"):
        # --run-local given in cli: do not skip local tests
        return
    skip_local = pytest.mark.skip(reason="need --run-local option to run")
    for item in items:
        if "local_model" in item.keywords:
            item.add_marker(skip_local)
