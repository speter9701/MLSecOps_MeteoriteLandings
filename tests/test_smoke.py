import meteorite


def test_package_has_version() -> None:
    assert meteorite.__version__ == "0.1.0"
