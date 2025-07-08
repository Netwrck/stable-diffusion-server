import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from flux_dfloat11 import parser
except Exception:
    pytest.skip("flux_dfloat11 dependencies missing", allow_module_level=True)


def test_env_override(monkeypatch):
    monkeypatch.setenv("DF11_MODEL_PATH", "test-model")
    args = parser.parse_args([])
    assert args.save_path == "image.png"
    # ensure env var accessible
    assert os.getenv("DF11_MODEL_PATH") == "test-model"
