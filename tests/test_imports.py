"""Test that all browser_service imports work"""
import pytest


def test_imports():
    """Test basic imports"""
    from browser_service import config
    from browser_service.config import BrowserServiceConfig
    assert config is not None


def test_version():
    """Test version is available"""
    from browser_service import __version__
    assert __version__ is not None
    assert isinstance(__version__, str)


if __name__ == "__main__":
    test_imports()
    test_version()
    print("All import tests passed!")
