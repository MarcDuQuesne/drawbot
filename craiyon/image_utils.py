from super_image import EdsrModel, ImageLoader
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Color:

    white = (255, 255, 255)
    red = (205, 0, 26)
    dark_red = (139, 0, 0)
    green = (0, 255, 23)
    dark_green = (0, 139, 10)

    blue = (46, 103, 248)
    light_blue = (13, 250, 230)

    yellow = (246, 250, 0)
    dark_yellow = (255, 205, 0)

    @classmethod
    def range(cls, _from, _to, steps):

        _from = np.array(_from)
        _to = np.array(_to)
        step = (_to - _from) / steps

        for i in range(steps):
            color = _from + i * step
            yield int(color[0]), int(color[1]), int(color[2])

    @classmethod
    def list(cls):
        colors = [
            v for v, m in vars(Color).items() if not (v.startswith("_") or callable(m))
        ]
        for color in colors:
            yield getattr(cls, color)

    @classmethod
    def color_couples(cls):

        couples = [
            (cls.yellow, cls.dark_yellow),
            (cls.red, cls.dark_red),
            (cls.green, cls.dark_green),
            (cls.light_blue, cls.blue),
        ]

        for element in couples:
            yield element


class ImageTransformer:
    @classmethod
    def enhance(cls, image, scale=4, output_file: Path = None):
        """
        Improve the resolution of an image with a scale factor.
        """

        model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)

        if isinstance(image, Path) or isinstance(image, str):
            image = Image.open(image)

        inputs = ImageLoader.load_image(image)
        pred = model(inputs)

        return ImageLoader._process_image_to_save(pred)

    @classmethod
    def quantize(cls, image, K=4, background_color_to=Color.white, output_file=None):
        """
        Group pixels in K clusters of color.
        """

        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(image)

        Z = image.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        # count the relative number of pixels per color
        unique, counts = np.unique(label.flatten(), return_counts=True)
        counts.sort()
        counts = counts[::-1][1:]
        logger.debug(f"Relative n of pixel per color: {counts / sum(counts) * 100}")

        # sets the background color (the most diffused color..) to background_color_to
        if background_color_to is not None:
            distances = ((center - background_color_to) ** 2).sum(axis=1)
            nearest_index = np.argmin(distances)
            center[nearest_index] = background_color_to

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        quantized_image = res.reshape((image.shape))

        return quantized_image, center

    @classmethod
    def extract_layers(cls, image, colors=None, background=Color.white):

        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(image)

        if colors is None:
            colors = np.unique(image.reshape(-1, image.shape[-1]), axis=0)

        layers = []
        for i, color in enumerate(colors):
            if np.array_equal(color, background):
                continue

            mask = cv2.inRange(image, color, color)
            masked = cv2.bitwise_and(image, image, mask=mask)
            black_pixels = np.where(
                (masked[:, :, 0] == 0) & (masked[:, :, 1] == 0) & (masked[:, :, 2] == 0)
            )
            # set those pixels to white
            masked[black_pixels] = background
            layers.append(masked)

        return layers


class ImageProcessor:
    def __init__(self, image):
        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(image)

        self.image = image

    @classmethod
    def external_contours(cls, image):

        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 200, 255, 0)
        return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    @staticmethod
    def remove_contours(image, contours, background_color=Color.white, width=5):
        _image = np.copy(image)
        return cv2.drawContours(_image, contours, -1, background_color, width)

    def compute_drawing_lines(self, pen_width=10):
        """
        Takes one of the quantized images, and computes a trajectory based on contours for the pen to follow, given a pen width (in pixels), so to fill each area with color.
        """

        # TODO the borders of the layers coincide.

        all_contours = []
        contours = [[1], [2]]
        _image = np.copy(self.image)
        while len(contours) > 1:
            contours, hierarchy = self.external_contours(_image)
            all_contours.append(contours)
            _image = self.remove_contours(_image, contours, width=pen_width)

        return all_contours

    def visualize_drawing_lines(
        self, contours_list, image=None, _from=Color.red, _to=Color.blue
    ):
        """
        Creates a visualization for the contours.
        """

        for contour, color in zip(
            contours_list, Color.range(_from, _to, len(contours_list))
        ):
            image = cv2.drawContours(image, contour, -1, color, 1)

        return image


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG, format="%(name)s.%(funcName)s | %(message)s"
    )

    # enhanced_image = ImageTransformer.enhance(
    #     image="images\dutch tile with insects-6.png"
    # )

    # quantized_image, colors = ImageTransformer.quantize(enhanced_image, K=4)
    # cv2.imwrite("images\kmeans3.png", quantized_image)

    # save_color_clusters(quantized_image, colors, output_folder="images")

    layers = ImageTransformer.extract_layers("images\kmeans3.png")
    for i, layer in enumerate(layers):
        cv2.imwrite(f"images\layer_{i}.png", layer)

    c_img = np.zeros(layers[0].shape, dtype=np.uint8)
    c_img.fill(255)
    for image, color in zip(layers, Color.color_couples()):
        processor = ImageProcessor(image)
        drawing_lines = processor.compute_drawing_lines()
        c_img = processor.visualize_drawing_lines(
            drawing_lines, _from=color[0], _to=color[1], image=c_img
        )
    cv2.imwrite("images\drawing_lines_3.png", c_img)
