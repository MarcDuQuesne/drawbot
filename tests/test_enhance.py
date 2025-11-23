
from turtle import color
import pytest
from craiyon.image_utils import ImageTransformer, ImageProcessor, Color
from pathlib import Path
import cv2
from PIL import Image
from svg_to_gcode.svg_parser import parse_file
from svg_to_gcode.compiler import Compiler, interfaces
from svg_to_gcode.formulas import linear_map
import numpy as np

IMAGES = Path(__file__).parent.parent / "images"
import logging

@pytest.fixture(autouse=True)
def setup_logger():
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all levels
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger().info("Logging initialized for test session.")
    yield
    logging.getLogger().info("Test session finished.")

@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "1.original" / "crab.png"),
        Path(IMAGES / "1.original" / "scarab.png"),
    ],
)
def test_enhance(imagepath: Path):

    scale = 4

    enhanced_image = ImageTransformer.enhance(
        image=imagepath,
        scale=scale,
    )

    root = imagepath.stem
    cv2.imwrite((IMAGES / "2.enhanced" / f"{root}_{scale}x.png").as_posix(), enhanced_image)

@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "2.enhanced" / "crab_4x.png"),
        Path(IMAGES / "2.enhanced" / "scarab_4x.png"),
    ],
)
def test_quantize(imagepath: Path):

    n_colors = 4
    root = imagepath.stem

    quantized_image, colors = ImageTransformer.quantize(imagepath, K=n_colors)
    cv2.imwrite((IMAGES / "3.quantized" / f"{root}_kmeans{n_colors}.png").as_posix(), quantized_image)

    layers = ImageTransformer.extract_layers(quantized_image)
    for i, layer in enumerate(layers):
        cv2.imwrite((IMAGES / "4.layered" / f"{root}_layer_{i}.png").as_posix(), layer)

@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "2.enhanced" / "crab_4x.png"),
        Path(IMAGES / "2.enhanced" / "scarab_4x.png"),
    ],
)
def test_quantize_to_palette(imagepath: Path):

    root = imagepath.stem
    # Load image (BGR)
    img = cv2.imread(imagepath.as_posix())

    # Define palette
    palette_bgr = np.array(Color.stabilo_88() + [Color.white], dtype=np.uint8)
    palette_bgr = np.array([Color.white, Color.black, Color.blue], dtype=np.uint8)

    quantized = ImageTransformer.quantize_to_palette(img, palette_bgr)

    cv2.imwrite((IMAGES / "3.quantized" / f"{root}_palette.png").as_posix(), quantized)

@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "2.enhanced" / "scarab_4x.png"),
    ],
)
def test_monocolor(imagepath: Path):

    root = imagepath.stem
    # Load image (BGR)
    img = cv2.imread(imagepath.as_posix())

    # Define palette
    monocolor = ImageTransformer.to_monocolor(img, Color.blue, background=Color.white)

    cv2.imwrite((IMAGES / "3.quantized" / f"{root}_monocolor.png").as_posix(), monocolor)


@pytest.mark.parametrize(
    "imagepath",
    [
        Path(IMAGES / "4.layered"),
    ],
)
def test_compute_drawing_lines(imagepath: Path):

        layers = [cv2.imread(image.as_posix()) for image in imagepath.glob("*.png")]

        colors = Color.list(); next(colors)  # skip white

        all_lines = layers[0].copy()
        all_lines.fill(255)

        for i, image in enumerate(layers):
            processor = ImageProcessor(image)
            drawing_lines = processor.compute_drawing_lines(pen_width=20)
            img = processor.visualize_drawing_lines(drawing_lines, color=next(colors))
            cv2.imwrite((IMAGES / "5.drawing_lines" / f"crab_lines_{i}.png").as_posix(), img)

            # combine all drawing lines
            all_lines = cv2.bitwise_and(all_lines, img)

        cv2.imwrite((IMAGES / "5.drawing_lines" / f"crab_lines_all.png").as_posix(), all_lines)

@pytest.mark.parametrize(
    "imagepath",
    [
        # Path(IMAGES / "3.quantized" / "crab"),
        Path(IMAGES / "3.quantized" / "scarab"),
    ],
)
def test_conversion(imagepath: Path):

    imagepath = [image for image in imagepath.glob("*.png")]
    for i, image in enumerate(imagepath):
        processor = ImageProcessor(image)
        root = Path(image).stem
        case = root.split("_")[0]
        processor.export_svg(filename=image, output_file=(IMAGES / "6.converted" / f"{case}" / f"{root}.svg"), turdsize=20)


class CustomInterface(interfaces.Gcode):
    def __init__(self):
        super().__init__()
        self.fan_speed = 1

    # Override the laser_off method such that it also powers off the fan.
    def laser_off(self):
        return "M107;\n" + "M5;"  # Turn off the fan + turn off the laser

    # Override the set_laser_power method
    def set_laser_power(self, power):
        if power < 0 or power > 1:
            raise ValueError(f"{power} is out of bounds. Laser power must be given between 0 and 1. "
                             f"The interface will scale it correctly.")

        return f"M106 S255\n" + f"M3 S{linear_map(0, 255, power)};"  # Turn on the fan + change laser power

    # Add pen up/down methods
@pytest.mark.parametrize(
    "image",
    [
       IMAGES / "6.converted" / "scarab" / "scarab_4x_monocolor.svg"
    ],
)
def test_svg_to_gcode(image:Path):
    """
    Test svg to gcode conversion with custom interface (experimental)
    """
    gcode_compiler = Compiler(CustomInterface, movement_speed=1000, cutting_speed=300, pass_depth=5)
    curves = parse_file(image.as_posix())
    root = image.stem
    gcode_compiler.append_curves(curves)
    gcode_compiler.compile_to_file((IMAGES / "7.gcode" / f"{root}.gcode").as_posix(), passes=1)