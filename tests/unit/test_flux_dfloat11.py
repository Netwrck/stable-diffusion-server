import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from flux_dfloat11 import parser
except Exception:  # pragma: no cover - optional dependency
    pytest.skip("flux_dfloat11 dependencies missing", allow_module_level=True)


def test_default_args():
    args = parser.parse_args([])
    assert args.prompt
    assert args.save_path == "image.png"
    assert args.seed == 0
    assert args.steps == 50
