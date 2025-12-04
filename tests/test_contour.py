
from craiyon.image_utils import HSVContourMapper
from pathlib import Path

from conftest import IMAGES

def test_contour_mapping():
    """
    Test the contour mapping functionality.
    """

    image = IMAGES / "1.original" / "crab.png"
    # Create a contour map from an image using the Value (brightness) dimension
    mapper = HSVContourMapper(image)

    for dimension in ['H', 'S', 'V']:
        mapper.generate_contour_map(Path(f"{image.stem}_contours_{dimension}.svg"), dimension=dimension, num_levels=10)

    # # Or use custom height matrix
    # height_matrix = mapper.create_height_matrix('S')  # Use Saturation instead
    # contours = mapper.compute_contours_from_height(height_matrix, num_levels=10)
    # mapper.export_contours_svg(contours, Path("saturation_map.svg"))