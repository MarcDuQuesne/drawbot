from fileinput import filename
from super_image import EdsrModel, ImageLoader
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import logging
import colorsys
from baffi.decorators.log_helpers import timeit
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY
from svgpathtools import svg2paths, smoothed_path, wsvg

logger = logging.getLogger(__name__)

class Color:

    # Opencv has BGR, not RGB

    white = (255, 255, 255)
    red = (40, 40, 200)
    dark_red = (0, 0, 139)
    green = (23, 255, 0)
    dark_green = (10, 139, 0)
    blue = (248, 103, 46)
    light_blue = (230, 250, 13)
    yellow = (20, 220, 250)
    dark_yellow = (0, 205, 255)
    black = (0, 0, 0)
    dark_yellow = (0, 235, 255)
    orange = (0, 120, 255)
    pink = (150, 90, 255)
    purple = (120, 40, 150)
    other_blue = (120, 80, 20)
    brown = (60, 80, 100)

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
            (cls.light_blue, cls.blue),
            (cls.yellow, cls.dark_yellow),
            (cls.red, cls.dark_red),
            (cls.green, cls.dark_green),
        ]

        for element in couples:
            yield element

    @classmethod
    def stabilo_88(cls):
        return [
            cls.yellow,
            cls.orange,
            cls.red,
            cls.pink,
            cls.purple,
            cls.blue,
            cls.green,
            cls.dark_green,
            cls.brown,
            cls.black,
        ]
    
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
            image = cv2.imread(image.as_posix())

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
    def quantize_to_palette(cls, image: np.ndarray, palette_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: HxWx3 uint8 image in BGR (OpenCV default)
        palette_bgr: Kx3 uint8 or float32 array of colors in BGR
        """

        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(image)

        # Ensure correct dtypes
        img = image.astype(np.int32)  # avoid overflow in subtraction
        palette = palette_bgr.astype(np.int32)

        h, w, c = img.shape
        pixels = img.reshape(-1, 3)  # (N, 3), N = H*W

        # Compute squared distances to each palette color
        # pixels[:, None, :] -> (N, 1, 3)
        # palette[None, :, :] -> (1, K, 3)
        # diff -> (N, K, 3)
        diff = pixels[:, None, :] - palette[None, :, :]  # broadcast
        dist2 = np.sum(diff * diff, axis=2)              # (N, K)

        # For each pixel, pick index of closest palette color
        nearest_idx = np.argmin(dist2, axis=1)           # (N,)

        # Map indices back to palette colors
        quantized_pixels = palette[nearest_idx]          # (N, 3)

        # Reshape to original image
        quantized_img = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
        return quantized_img

    @classmethod
    def extract_palette(cls, image, K=4):
        """
        Extract the K most prominent colors from an image.
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

        return np.uint8(center)

    @classmethod
    def to_monocolor(cls, image, color=Color.black, threshold=128, background=Color.white):
        """
        Convert an image to black and white based on a threshold.
        """

        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        if color != Color.black:
            colored_image = np.zeros_like(image)
            colored_image[bw_image == 0] = color

        colored_image[bw_image == 255] = background
        return colored_image

    @classmethod
    def extract_layers(cls, image, colors=None, background=Color.white):

        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(image)

        if colors is None:
            colors = np.unique(image.reshape(-1, image.shape[-1]), axis=0)

        def lighness(rgb):
            return colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])[1]

        # we sort colors (and thus layers)
        # by lightness. We want to write light layers first.
        colors = sorted(colors, key=lighness)

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
    def __init__(self, image = None):

        if isinstance(image, Path) or isinstance(image, str):
            logger.info(f"Reading {image}.")
            image = cv2.imread(image)

        self.image = image

    @classmethod
    def external_contours(cls, image):

        # # We assume the image has only one color and a white background.
        # # Ensure image is BGR (drop alpha channel if present)
        # if image.ndim == 3 and image.shape[2] == 4:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # # Flatten pixels and find unique colors
        # pixels = image.reshape(-1, 3)
        # unique_colors = np.unique(pixels, axis=0)

        # # Consider near-white as background to tolerate compression artifacts
        # white_thresh = 250
        # is_white = np.all(unique_colors >= white_thresh, axis=1)
        # white_colors = unique_colors[is_white]
        # non_white_colors = unique_colors[~is_white]

        # # Validate presence of white background
        # if white_colors.size == 0:
        #     raise ValueError("Image must have a white background (no near-white pixels found)")

        # # Validate there is exactly one non-background color
        # if non_white_colors.size == 0:
        #     raise ValueError("Image contains only white background (no colored pixels found)")
        # if non_white_colors.shape[0] > 1:
        #     logger.debug(f"Unique non-white colors found: {non_white_colors}")
        #     raise ValueError(
        #     f"Image must contain exactly one non-background color. Found {non_white_colors.shape[0]} distinct non-white colors."
        #     )

        # convert to CV_8UC1 images

        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 254, 255, 0)
        # # The function cv::findContours describes the contour of areas consisting of ones.
        # # The areas in which we are interested are black, though.
        thresh = 255 - thresh
        return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    @staticmethod
    def remove_contours(image, contours, background_color=Color.white, width=5):
        _image = np.copy(image)
        return cv2.drawContours(_image, contours, -1, background_color, width)

    @timeit
    def compute_drawing_lines(self, pen_width, smoothing_factor=0.001):
        """
        Takes one of the quantized images, and computes a trajectory based on contours for the pen to follow, given a pen width (in pixels), so to fill each area with color.
        """

        logger.info(f"Computing drawing lines with width: {pen_width} pixels.")
        logger.debug(f"Smoothing factor {smoothing_factor}")

        # TODO the borders of the layers coincide.

        all_contours = []
        contours = [[1], [2]]
        _image = np.copy(self.image)
        while len(contours) > 1:
            contours, hierarchy = self.external_contours(_image)
            contours = [
                self.smooth_contour(contour, smoothing_factor) for contour in contours
            ]
            all_contours.append(contours)
            _image = self.remove_contours(_image, contours, width=pen_width)

        return all_contours

    def smooth_contour(self, contour, smoothing_factor=0.001):
        # smooth contour
        epsilon = smoothing_factor * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)

    def visualize_drawing_lines(
        self, contours_list, image=None, color=Color.black, pen_width=1
    ):
        """
        Creates a visualization for the contours.
        """

        if image is None:
            image = np.zeros(self.image.shape, dtype=np.uint8)
            image.fill(255)

        for contour in contours_list:
            image = cv2.drawContours(image, contour, -1, color, 1)

        return image

    def export_svg(self, filename: str, output_file: Path):

        try:
            image = Image.open(filename)
        except IOError:
            print("Image (%s) could not be loaded." % filename)
            return
        bm = Bitmap(image, blacklevel=1)
        # bm.invert()
        plist = bm.trace(
            turdsize=2,
            turnpolicy=POTRACE_TURNPOLICY_MINORITY,
            alphamax=1,
            opticurve=False,
            opttolerance=0.2,
        )
        with open(output_file.as_posix(), "w") as fp:
            fp.write(
                f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">''')
            parts = []
            for curve in plist:
                fs = curve.start_point
                parts.append(f"M{fs.x},{fs.y}")
                for segment in curve.segments:
                    if segment.is_corner:
                        a = segment.c
                        b = segment.end_point
                        parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                    else:
                        a = segment.c1
                        b = segment.c2
                        c = segment.end_point
                        parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
                parts.append("z")
            fp.write(f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>')
            fp.write("</svg>")
        logger.info(f"SVG saved to {filename}.svg")

    def optimize_svg(self, input_file: Path, output_file: Path):
        """
        Optimize SVG file size.
        """
        # Load SVG paths
        paths, attributes = svg2paths(input_file.as_posix())

        # Smooth each path
        smoothed_paths = [smoothed_path(path) for path in paths if path.iscontinuous()]

        # Save the smoothed paths to a new SVG file
        wsvg(smoothed_paths, filename=output_file.as_posix())



