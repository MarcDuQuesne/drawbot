import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)6s %(name)12s %(funcName)30s %(message)s"
)

logging.getLogger("h5py").setLevel(logging.WARNING)

from image_utils import ImageProcessor
from gcode_utils import GCodeSender, GCodeTransformer
import cv2

logger = logging.getLogger("drawbot")


def process_image(image="tests\\square.png"):
    logger.info(f"Opening image {image}")
    image = cv2.imread(image)

    desired_image_size = 30  # mm
    pixel_per_mm = image.shape[0] / desired_image_size

    logger.info(f"Desired Image size: {desired_image_size} mm.")
    logger.debug(f"pixel per mm: {pixel_per_mm}.")

    drawing_lines = ImageProcessor(image).compute_drawing_lines()

    msg = GCodeTransformer().contours_to_gcode(drawing_lines, scale=pixel_per_mm)
    with open("test.gcode", "w") as f:
        f.write("\n".join(msg))


if __name__ == "__main__":
    process_image()

    # sender = GCodeSender(port='COM3')
    # sender.send(msg)
