import pytest


@pytest.fixture(autouse=True)
def disable_project_dotenv(monkeypatch):
    monkeypatch.setenv("AI_PROFESSOR_DISABLE_DOTENV", "1")
