# save images in 1-10 compresion timing the results
import pytest
pytest.skip(reason="script file, not a test", allow_module_level=True)

from pathlib import Path
from time import time
def test_compression():
    save_dir = Path("./imgs-sd/test/")
    save_dir.mkdir(exist_ok=True, parents=True)

    from PIL import Image

    image = Image.open("/home/lee/code/sdif/imgs-sd/Woody.png").convert("RGB")
    start = time()

    image.save(save_dir / f"woody-.webp", format="webp")
    end = time()
    print(f"Time to save image with quality : {end - start}")

    for i in range(0, 100):
        start = time()

        image.save(save_dir / f"woody-{i}.webp", quality=i, optimize=True, format="webp")
        end = time()
        print(f"Time to save image with quality {i}: {end - start}")
