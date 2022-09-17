from PIL import Image
from io import BytesIO
from base64 import b64decode
import requests
import json
from pathlib import Path
import logging


logger = logging.getLogger("CrayionAPI")


class CrayionAPI:

    URL = "https://backend.craiyon.com/generate"

    ILLEGALCHARS = r'<>:"/\|?*'.join([chr(i) for i in range(32)])
    MAX_PROMPT_LEGTH = 50

    @classmethod
    def get_images(cls, prompt: str):

        logger.info(f'Using prompt: "{prompt}"')

        if (
            any([char in prompt for char in CrayionAPI.ILLEGALCHARS])
            or len(prompt) > CrayionAPI.MAX_PROMPT_LEGTH
        ):
            raise RuntimeError("Not a valid prompt.")

        response = requests.post(
            CrayionAPI.URL,
            headers={"Content-Type": "application/json"},
            data=f'{{"prompt": "{prompt}<br>"}}',
        )

        response.raise_for_status()
        payload = json.loads(response.text)

        return [BytesIO(b64decode(image)) for image in payload["images"]]


class ImageUtil:
    @classmethod
    def save_images(cls, images: list, format: str, base_name: str, directory: Path):

        logger.info(f"Saving images to {directory} in format {format}.")

        for i, image in enumerate(images):
            im = Image.open(image)
            fileName = f"{base_name}-{i}.{format}"
            im.save(directory / fileName)


def call_api():

    import argparse

    logging.basicConfig(
        level=logging.DEBUG, format="%(name)s.%(funcName)s | %(message)s"
    )

    parser = argparse.ArgumentParser(description="Crayion API.")
    parser.add_argument(
        "-p",
        "--prompt",
        action="store",
        type=str,
        help="The text used to create the images.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        action="store",
        type=Path,
        help="Output folder",
        default=Path.cwd(),
    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        raise RuntimeError(f"{args.output_dir} does not exist.")

    images = CrayionAPI.get_images(prompt=args.prompt)
    ImageUtil.save_images(
        images, format="png", base_name=args.prompt, directory=args.output_dir
    )


if __name__ == "__main__":
    call_api()
