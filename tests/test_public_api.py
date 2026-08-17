from tagatlas import Localizer, TagMap, __version__


def test_package_imports():
    assert Localizer is not None
    assert TagMap is not None
    assert __version__ == "0.1.0"
