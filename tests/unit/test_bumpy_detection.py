from pathlib import Path

from PIL import Image
from loguru import logger

from stable_diffusion_server.bumpy_detection import detect_too_bumpy

current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data"


def test_detect_too_bumpy():
    files = [
        "bug3.webp",
        "bug4.webp",
    ]
    for file in files:
        image = Image.open(data_dir / file)
        is_bumpy = detect_too_bumpy(image)
        assert is_bumpy

    image = Image.open(
        data_dir /
        "Serqet-Selket-goddess-of-protection-Egyptian-Heritage-octane-render-cinematic-color-grading-soft-light-atmospheric-reali.png"
    )
    is_bumpy = detect_too_bumpy(image)
    assert not is_bumpy

    # run over every img in outputs dir
    outputs_dir = data_dir.parent / "outputs"
    if outputs_dir.exists():
        for file in outputs_dir.iterdir():
            if file.is_file():
                image = Image.open(file)
                is_bumpy = detect_too_bumpy(image)
                assert not is_bumpy

    # run over every dir in tests/data/bugs dir
    bugs_dir = data_dir / "bugs"
    logger.info("checking bugs dir")
    for file in bugs_dir.iterdir():
        if file.is_file():
            image = Image.open(file)
            logger.info(f"checking {file}")
            is_bumpy = detect_too_bumpy(image)
            assert is_bumpy
