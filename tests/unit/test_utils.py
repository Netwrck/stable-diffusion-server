from stable_diffusion_server import utils
from loguru import logger


def test_log_time():
    messages = []
    sink_id = logger.add(lambda m: messages.append(m), format="{message}")
    try:
        with utils.log_time("test"):
            pass
    finally:
        logger.remove(sink_id)

    assert any("start" in m for m in messages)
    assert any("end" in m for m in messages)
