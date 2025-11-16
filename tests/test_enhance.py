    
import pytest
from craiyon.image_utils import ImageTransformer, ImageProcessor, Color
from pathlib import Path
import cv2
from PIL import Image
  
IMAGES = Path(__file__).parent.parent / "images"

@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "1.original" / "crab.png"),
    ],
) 
def test_enhance(imagepath: Path):
    
    scale = 4

    enhanced_image = ImageTransformer.enhance(
        image=imagepath,
        scale=scale,
    )

    cv2.imwrite((IMAGES / "2.enhanced" / f"crab_{scale}x.png").as_posix(), enhanced_image) 

@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "2.enhanced" / "crab_4x.png"),
    ],
) 
def test_quantize(imagepath: Path):

    n_colors = 10

    quantized_image, colors = ImageTransformer.quantize(imagepath, K=n_colors)
    cv2.imwrite((IMAGES / "3.quantized" / f"crab_kmeans{n_colors}.png").as_posix(), quantized_image)

    layers = ImageTransformer.extract_layers(quantized_image)
    for i, layer in enumerate(layers):
        cv2.imwrite((IMAGES / "4.layered" / f"crab_layer_{i}.png").as_posix(), layer)


@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "4.layered"),
    ],
) 
def test_compute_drawing_lines(imagepath: Path):

        for i, image in enumerate(imagepath.glob("*.png")):
            processor = ImageProcessor(image)
            drawing_lines = processor.draw_drawing_lines(pen_width=1)
            cv2.imwrite((IMAGES / "5.drawing_lines" / f"crab_lines_{i}.png").as_posix(), drawing_lines)

@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "4.layered"),
    ],
) 
def test_compute_all_drawing_lines(imagepath: Path):

        layers = [cv2.imread(image.as_posix()) for image in imagepath.glob("*.png")]

        processor = ImageProcessor(layers[0])
        drawing_lines = processor.draw_all_drawing_lines(layers=layers, pen_width=1)
        cv2.imwrite((IMAGES / "5.drawing_lines" / f"crab_path.png").as_posix(), drawing_lines)