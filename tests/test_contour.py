
from craiyon.contours import HSVContourMapper, Pipeline, CountourFilter
from pathlib import Path
import cv2
import numpy as np

from conftest import IMAGES
import pytest 

@pytest.mark.parametrize(
    "dimension, method",
    [
        ('V', 'opencv'),
        ('V', 'matplotlib'),
    ],
)
def test_contour_mapping(dimension, method):
    """
    Test the contour mapping functionality.
    """

    image = IMAGES / "1.original" / "crab.png"
    # Create a contour map from an image using the Value (brightness) dimension
    mapper = HSVContourMapper(image)

    mapper.hsv_image = cv2.GaussianBlur(mapper.hsv_image,(7,7),0)
    mapper.hsv_image = cv2.GaussianBlur(mapper.hsv_image,(7,7),0)

    contour_levels = mapper.generate_contour_map(dimension=dimension, num_levels=10, method=method)

    filtering_pipeline = Pipeline(
        CountourFilter(lambda c: cv2.contourArea(c.astype(np.float32)) > 10),  # filter small contours
        CountourFilter(lambda c: len(c) >= 2), # filter too short contours
        CountourFilter(lambda c: cv2.arcLength(c.astype(np.float32), False) > 20)
    )

    contour_levels = filtering_pipeline(contour_levels)
    optimized_contours_levels = mapper.optimize_order_contours(contour_levels)

    mapper.export_contours_svg(optimized_contours_levels, IMAGES / "6.converted" / f"{image.stem}_{dimension}_{method}.svg")
