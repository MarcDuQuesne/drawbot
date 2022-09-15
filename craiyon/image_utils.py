from super_image import EdsrModel, ImageLoader
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

WHITE = np.array([255, 255, 255])


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

        if output_file:
            ImageLoader.save_image(pred, output_file)

        return ImageLoader._process_image_to_save(pred)

    @classmethod
    def quantize(cls, image, K=4, background_color_to=WHITE, output_file=None):
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
        unique, counts = np.unique(label.flatten(), return_counts=True)
        counts.sort()
        counts = counts[::-1][1:]
        logger.debug(f"Relative n of pixel per color: {counts / sum(counts) * 100}")

        if background_color_to is not None:
            distances = ((center - background_color_to) ** 2).sum(axis=1)
            nearest_index = np.argmin(distances)
            center[nearest_index] = background_color_to

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        quantized_image = res.reshape((image.shape))

        if output_file:
            cv2.imwrite("kmeans3.png", output_file)

        return quantized_image, center


def save_color_clusters(image, colors, background=WHITE, output_folder=None):

    color_clusters = []
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

        if output_folder:
            cv2.imwrite(output_folder / f"{i}.png", masked)
        color_clusters.append(mask)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG, format="%(name)s.%(funcName)s | %(message)s"
    )

    enhanced_image = ImageTransformer.enhance(
        image="images\dutch tile with insects-6.png"
    )

    quantized_image, colors = ImageTransformer.quantize(enhanced_image, K=4)
    cv2.imwrite("images\kmeans3.png", quantized_image)
