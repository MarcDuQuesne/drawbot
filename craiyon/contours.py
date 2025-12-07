""""
HSV Contour Mapper
Generates contour lines from an image based on HSV color space dimensions.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from typing import Literal
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
logger = logging.getLogger(__name__)


class Pipeline:
    """
    A simple pipeline to chain multiple processing steps.
    """

    def __init__(self, *filters):
        self.filters = filters

    def __call__(self, items):
        for flt in self.filters:
            items = flt(items)
        return list(items)

class CountourFilter:
    """
    Filter contours.
    """ 
    def __init__(self, criteria: callable):
        self.criteria = criteria

    def __call__(self, contour_levels):
            for level_idx, (level, contours) in enumerate(contour_levels):
                filtered_contours = [cnt for cnt in contours if self.criteria(cnt)]
                logger.debug(f"Level {level:.1f}: {len(contours)} -> {len(filtered_contours)} f{self.criteria.__name__} contours after filtering.")
                yield (level, filtered_contours)

class HSVContourMapper:
    """
    Converts an image to HSV and generates contour lines based on one HSV dimension,
    similar to a topographic map where elevation is determined by H, S, or V values.
    """
    
    def __init__(self, image,blur: bool = True):
        if isinstance(image, Path) or isinstance(image, str):
            logger.info(f"Reading {image}.")
            image = cv2.imread(image)
        

        if blur:
            logger.info("Applying Gaussian blur to the image.")
            image = cv2.GaussianBlur(image, (5, 5), 0)

        self.image = image
        self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        logger.info("Image converted to HSV color space.")
    
    def create_height_matrix(self, dimension='V'):
        """
        Create a 2D height matrix from one HSV dimension.
        
        Args:
            dimension: 'H' (Hue), 'S' (Saturation), or 'V' (Value)
        
        Returns:
            2D numpy array where each pixel value represents the height
        """
        dimension_map = {'H': 0, 'S': 1, 'V': 2}
        if dimension not in dimension_map:
            raise ValueError(f"Dimension must be 'H', 'S', or 'V', got {dimension}")
        
        idx = dimension_map[dimension]
        height_matrix = self.hsv_image[:, :, idx]
        logger.info(f"Created height matrix from HSV {dimension} dimension.")

        return height_matrix
    
    def compute_contours_from_height(self, height_matrix, num_levels=10):
        """
        Compute contour lines at different height levels.
        
        Args:
            height_matrix: 2D array of height values
            num_levels: Number of contour levels to extract
        
        Returns:
            List of contour levels and their corresponding OpenCV contours
        """
        height_normalized = ((height_matrix - height_matrix.min()) / 
                            (height_matrix.max() - height_matrix.min()) * 255).astype(np.uint8)
        
        contour_levels = []
        levels = np.linspace(0, 255, num_levels + 2)[1:-1]
        
        for level in levels:
            _, binary = cv2.threshold(height_normalized, int(level), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contour_levels.append((level, contours))
            logger.debug(f"Found {len(contours)} contours at level {level:.1f}")
        
        return contour_levels
    
    def compute_contours_matplotlib(self, height_matrix, num_levels=10):
        """
        Compute contour lines using matplotlib's contour algorithm.
        
        Args:
            height_matrix: 2D array of height values
            num_levels: Number of contour levels to extract
        
        Returns:
            List of contour levels and their corresponding matplotlib ContourSet
        """
        # Normalize height matrix to 0-255 range for consistent levels
        height_matrix = gaussian_filter(height_matrix, sigma=1)
        height_normalized = (height_matrix - height_matrix.min()) / (height_matrix.max() - height_matrix.min())
        
        # Create meshgrid for contour computation
        h, w = height_matrix.shape
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        # Compute contours using matplotlib
        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
        contour_set = ax.contour(X, Y, height_normalized, levels=num_levels)
        
        # Extract contour data
        contours_per_level = []  # list of (level_value, [contours])
        for level_value, segs in zip(contour_set.levels, contour_set.allsegs):
            # segs is a list of (N, 2) arrays: one per contour at this level
            level_contours = [seg for seg in segs if len(seg) >= 2]
            contours_per_level.append((level_value, level_contours))
            logger.debug(f"Found {len(level_contours)} contours at level {level_value:.3f}")
        plt.close(fig)
        return contours_per_level
    
    def generate_contour_map(self, dimension='V', num_levels=5, method: Literal['opencv', 'matplotlib'] = 'opencv'):
        """
        Generate contour map with choice of algorithm.
        
        Args:
            output_file: Path to save SVG
            dimension: 'H', 'S', or 'V' (which HSV dimension determines height)
            num_levels: Number of contour levels
            method: 'opencv' or 'matplotlib' - which contour algorithm to use
        """
        if method not in ['opencv', 'matplotlib']:
            raise ValueError(f"Method must be 'opencv' or 'matplotlib', got {method}")
        
        height_matrix = self.create_height_matrix(dimension)
        
        if method == 'opencv':
            contour_levels = self.compute_contours_from_height(height_matrix, num_levels)
            logger.info(f"Contour map generated using OpenCV with HSV {dimension} dimension and {num_levels} levels.")
        else:  # matplotlib
            contour_levels = self.compute_contours_matplotlib(height_matrix, num_levels)
            logger.info(f"Contour map generated using matplotlib with HSV {dimension} dimension and {num_levels} levels.")

        return contour_levels

    def smoothen_contours(self, contour, sigma=1.0):
        """
        Smoothen contour lines using Gaussian filter.
        
        Args:
            contours: List of contours (each contour is an Nx1x2 array)
            sigma: Standard deviation for Gaussian kernel
        
        Returns:
            List of smoothened contours
        """
        # smoothened_contours = []
        # for contour in contours:
        #     if len(contour) < 3:
        #         smoothened_contours.append(contour)
        #         continue
            
        #     x = contour[:, 0, 0]
        #     y = contour[:, 0, 1]
            
        #     x_smooth = gaussian_filter(x, sigma=sigma)
        #     y_smooth = gaussian_filter(y, sigma=sigma)
            
        #     smoothened_contour = np.stack((x_smooth, y_smooth), axis=-1).reshape(-1, 1, 2).astype(np.int32)
        #     smoothened_contours.append(smoothened_contour)
        
        # logger.info(f"Smoothened {len(contours)} contours with sigma={sigma}.")
        # return smoothened_contours
    
        x,y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x,y], u=None, s=1.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]

        return np.asarray(res_array, dtype=np.int32)

    def optimize_order_contours(self, contour_levels):
        """
        Optimize the order of contours to minimize travel distance.
        
        Args:
            contour_levels: List of (level, contours) tuples
        
        Returns:
            Optimized list of (level, contours) tuples
        """
        optimized_levels = []
        
        for level, contours in contour_levels:
            if not contours:
                optimized_levels.append((level, contours))
                continue
            
            ordered = [contours[0]]
            remaining = contours[1:]
            
            current_point = ordered[-1][-1]  # End point of the last added contour
            
            while remaining:
                # Find the closest contour start point
                distances = [np.linalg.norm(current_point - cnt[0]) for cnt in remaining]
                min_idx = np.argmin(distances)
                
                next_contour = remaining.pop(min_idx)
                ordered.append(next_contour)
                current_point = ordered[-1][-1]
            
            optimized_levels.append((level, ordered))
            logger.info(f"Optimized order of {len(contours)} contours at level {level:.1f}.")
        
        return optimized_levels

    def export_contours_svg(self, contour_levels, output_file: Path, stroke_width: int = 1):
        """
        Export contour lines as SVG, with different opacity/color for different levels.
        
        Args:
            contour_levels: List of (level, contours) tuples from compute_contours_from_height
            output_file: Path to save SVG
            stroke_width: Width of contour lines in SVG
        """
        h, w = self.hsv_image.shape[:2]

        # Minimize travel distance by optimizing contour order, per level.
        contour_levels = self.optimize_order_contours(contour_levels)

        with open(output_file.as_posix(), "w") as fp:
            fp.write(f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">''')
            fp.write(f'<rect width="{w}" height="{h}" fill="white"/>\n')

            for level_idx, (level, contours) in enumerate(contour_levels):
                # Vary opacity based on level (higher levels more opaque)
                # opacity = 0.3 + (level_idx / len(contour_levels)) * 0.7
                opacity = 1
                # Vary color from light to dark
                gray_value = int(50 + (level_idx / len(contour_levels)) * 200)
                color = f"rgb({gray_value},{gray_value},{gray_value})"
                
                for contour in contours:
                    # Build SVG path from contour points
                    path_parts = []
                    for i, point in enumerate(contour):
                        x, y = point.flatten()
                        if i == 0:
                            path_parts.append(f"M{x},{y}")
                        else:
                            path_parts.append(f"L{x},{y}")
                    path_parts.append("Z")
                    
                    path_data = "".join(path_parts)
                    fp.write(f'<path d="{path_data}" stroke="{color}" fill="none" stroke-width="{stroke_width}" opacity="{opacity}"/>\n')
            
            fp.write("</svg>")
        
        logger.info(f"Contour SVG exported to {output_file}")