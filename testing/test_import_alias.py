import importlib

import skytropd


def test_skytropd_import():
    assert hasattr(skytropd, "__version__")
    assert hasattr(skytropd, "TropD_Metric_PSI")


def test_skytropd_submodule_imports():
    tutorial = importlib.import_module("skytropd.tutorial")
    metrics = importlib.import_module("skytropd.metrics")

    assert hasattr(tutorial, "lat")
    assert hasattr(metrics, "TropD_Metric_PSI")
