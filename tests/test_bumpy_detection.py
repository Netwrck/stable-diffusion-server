from pathlib import Path

from PIL import Image
from loguru import logger

from stable_diffusion_server.bumpy_detection import detect_too_bumpy

current_dir = Path(__file__).parent
def test_detect_too_bumpy():
    files = [
        # "data/bug.webp",
        # "data/bug1.webp",
        # "data/bug2.webp",
        "data/bug3.webp",
        "data/bug4.webp",
    ]
    for file in files:
        image = Image.open(current_dir /f'{file}')
        is_bumpy = detect_too_bumpy(image)
        assert is_bumpy == True

    image = Image.open(current_dir / "data/Serqet-Selket-goddess-of-protection-Egyptian-Heritage-octane-render-cinematic-color-grading-soft-light-atmospheric-reali.png")
    is_bumpy= detect_too_bumpy(image)
    assert is_bumpy == False

    # run over every img in outputs dir
    outputs_dir = (current_dir).parent / "outputs"
    for file in outputs_dir.iterdir():
        if file.is_file():
            image = Image.open(file)
            is_bumpy = detect_too_bumpy(image)
            assert is_bumpy == False


    # run over every dir in tests/data/bugs dir
    bugs_dir = current_dir / "data/bugs"
    logger.info("checking bugs dir")
    for file in bugs_dir.iterdir():
        if file.is_file():
            image = Image.open(file)
            logger.info(f"checking {file}")
            is_bumpy = detect_too_bumpy(image)
            assert is_bumpy == True
