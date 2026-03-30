from pathlib import Path


def test_api_package_exists() -> None:
    assert Path("src/api/__init__.py").is_file()
