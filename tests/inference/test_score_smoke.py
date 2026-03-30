from pathlib import Path


def test_inference_package_exists() -> None:
    assert Path("src/inference/__init__.py").is_file()
