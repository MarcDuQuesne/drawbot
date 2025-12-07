from craiyon.hatched import hatch
from craiyon.hatched_cv2 import Hatcher
from conftest import IMAGES

import cv2
import numpy as np

def test_hatch():
    """
    Test the hatch function.
    """
    input = IMAGES / "1.original" / "crab.png"
    output = IMAGES / "8.hatched" / f"{input.stem}_hatched.svg"

    hatch(
        file_path=input.as_posix(),
        levels=(64, 128, 192),
        save_svg=False,
        circular=True,
        hatch_pitch=3,
    )

def test_cv2_hatcher():
    """
    Test the Hatcher class using OpenCV.
    """
    input = IMAGES / "1.original" / "crab.png"

    hatcher = Hatcher(image_path=input)
    hatched_images = hatcher.hatch(thresholds=[64,128, 192], method="circular")

    h, w = hatcher.image.shape[:2]
    final_result = np.full((h, w), 255, dtype=np.uint8)  # white background
    # combine all the images in one:
    for hatched in hatched_images:
        final_result = cv2.bitwise_and(final_result, hatched)
    
    cv2.imwrite("test_hatched_output.png", final_result)