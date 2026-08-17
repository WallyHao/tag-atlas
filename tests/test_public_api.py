from tagatlas import Localizer, __version__


def test_package_imports():
    assert Localizer is not None
    assert __version__ == "0.1.0"
