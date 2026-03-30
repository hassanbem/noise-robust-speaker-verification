from pathlib import Path


def test_ui_package_exists() -> None:
    assert Path("src/ui/__init__.py").is_file()
