import importlib
import sys
import os
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATORS_DIR = os.path.join(TESTS_DIR, "generate_julia_data")

_GENERATOR_MAP = {
    "test_lensing": "generate_lensing",
    "test_logpdf": "generate_logpdf",
    "test_gradients": "generate_gradients",
    "test_wiener_filter": "generate_wiener_filter",
    "test_map_joint": "generate_map_joint",
    "test_simulated_cls": "generate_simulated_cls",
    "test_covariance_matrices": "generate_covariance_matrices",
}


def pytest_addoption(parser):
    parser.addoption(
        "--generate",
        action="store_true",
        default=False,
        help="Run Julia data generators before tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--generate"):
        return
    modules_needed = set()
    for item in items:
        mod_name = item.module.__name__.rsplit(".", 1)[-1]
        if mod_name in _GENERATOR_MAP:
            modules_needed.add(mod_name)
    if not modules_needed:
        return
    if GENERATORS_DIR not in sys.path:
        sys.path.insert(0, GENERATORS_DIR)
    from _preamble import init_julia #type: ignore
    jl = init_julia()
    for mod_name in sorted(modules_needed):
        gen_module_name = _GENERATOR_MAP[mod_name]
        print(f"\n=== Running generator: {gen_module_name} ===")
        gen = importlib.import_module(gen_module_name)
        gen.run(jl)
