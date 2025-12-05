from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SVGContourMapper:

    

    def export_contours_svg(self, contour_levels: list, output_file: Path, stroke_width:int=1):
            """
            Export contour lines as SVG, with different opacity/color for different levels.
            
            Args:
                contour_levels: List of (level, contours) tuples from compute_contours_from_height
                output_file: Path to save SVG
                stroke_width: Width of contour lines in SVG
            """
            h, w = self.hsv_image.shape[:2]
            
            with open(output_file.as_posix(), "w") as fp:
                fp.write(f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">''')
                fp.write(f'<rect width="{w}" height="{h}" fill="white"/>\n')
                
                for level_idx, (level, contours) in enumerate(contour_levels):
                    # Vary opacity based on level (higher levels more opaque)
                    opacity = 0.3 + (level_idx / len(contour_levels)) * 0.7
                    # Vary color from light to dark
                    gray_value = int(50 + (level_idx / len(contour_levels)) * 200)
                    color = f"rgb({gray_value},{gray_value},{gray_value})"
                    
                    for contour in contours:
                        if len(contour) < 2:
                            continue
                        
                        if self.area_of_contour(contour) < 10:
                            continue  # skip tiny contours
                        # contour = self.smoothen_contours(contour)

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