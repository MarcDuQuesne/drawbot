
from craiyon.contours import HSVContourMapper
from pathlib import Path

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
    mapper.generate_contour_map(Path(f"{image.stem}_contours_{dimension}_{method}.svg"), dimension=dimension, num_levels=10, method=method)
