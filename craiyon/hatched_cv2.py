from pathlib import Path
import cv2
import numpy as np
from typing import List

from typing import Tuple, Sequence, Optional, Literal
from cv2.typing import MatLike

import math
import random

Contours = Tuple[Sequence[MatLike], MatLike]

class Hatcher:
    
    def __init__(self, image_path: Path):
        """Initialize the Hatcher class."""
        self.image: np.ndarray = cv2.imread(str(image_path))
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


    def find_contours(self, min_threshold: int, max_threshold: int) -> Contours:
        """Find contours in the image based on the given threshold."""
        _, binary_image = cv2.threshold(self.gray_image, min_threshold, max_threshold, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def draw_linear_hatches_mask(self, hatch_pitch: int, angle: Optional[float] = None) -> np.ndarray:
        """Draw hatch patterns at a given angle (degrees). If angle is None, a random angle is chosen."""
        h, w = self.image.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)  # white background

        if angle is None:
            angle = random.uniform(0.0, 180.0)
        # convert degrees to radians
        theta = math.radians(angle)

        # direction vector (unit) along the line, and perpendicular (normal) to step lines
        dir_vec = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        norm_vec = np.array([-dir_vec[1], dir_vec[0]], dtype=float)

        # center and length to draw long lines that fully cross the image
        center = np.array([w / 2.0, h / 2.0], dtype=float)
        diag = math.hypot(w, h)
        L = diag * 1.5

        # choose offsets along the normal to cover the whole image
        cover_dist = diag + max(h, w)
        start_offset = -cover_dist
        end_offset = cover_dist
        # ensure at least one line if hatch_pitch is larger than cover_dist
        if hatch_pitch <= 0:
            hatch_pitch = 1
        offsets = np.arange(start_offset, end_offset + 1, hatch_pitch)

        for t in offsets:
            p0 = center + norm_vec * float(t)
            p1 = p0 - dir_vec * L
            p2 = p0 + dir_vec * L
            cv2.line(mask, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), 0, 1)

        return mask
    
    def draw_circular_hatches_mask(self, hatch_pitch: int) -> np.ndarray:
        """Draw circular hatch patterns."""
        h, w = self.image.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)  # white background

        center = (w // 2, h // 2)
        max_radius = int(math.hypot(w / 2, h / 2)) + hatch_pitch

        for radius in range(hatch_pitch, max_radius, hatch_pitch):
            cv2.circle(mask, center, radius, 0, 1)

        return mask


    def hatch(self, 
              thresholds: List[int],
              method: Literal["linear", "circular"] = "linear",
              hatch_pitch: Optional[List[int]] = None) -> List[np.ndarray]:
        """Apply hatching to the image based on the given thresholds."""
        final_image = np.zeros_like(self.image)
        final_image.fill(255)  # White background
        
        if not hatch_pitch:
            hatch_pitch = [int(threshold / 255 * 10) for threshold in thresholds]

        levels = []
        thresholds.append(255)  # ensure the last threshold is 255
        for i in range(len(thresholds) - 1):

            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, thresholds[i], thresholds[i+1], cv2.THRESH_BINARY)            

            # angle may be passed later; by default draw_linear_hatches_mask will pick a random angle
            if method == "linear":
                hatch_mask = self.draw_linear_hatches_mask(hatch_pitch[i])
            elif method == "circular":
                hatch_mask = self.draw_circular_hatches_mask(hatch_pitch[i])
            else:
                raise ValueError(f"Unknown hatching method: {method}")

            # white where both are black
            levels.append(cv2.bitwise_or(binary_image, hatch_mask))

        return levels