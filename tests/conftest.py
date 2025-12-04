from pathlib import Path
import logging

import pytest
IMAGES = Path(__file__).parent.parent / "images"


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    logging.basicConfig(level=logging.DEBUG)