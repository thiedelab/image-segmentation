import numpy as np
from typing import Tuple, Callable
from math import floor, ceil
from em_analysis.contour_analysis import extract_long_contours_from_movie
from tqdm import tqdm

def _find_points_in_field_of_view(
    contour: np.ndarray,
    half_patch_size: int,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    far_enough_right = contour[:, 1] > half_patch_size
    far_enough_left = contour[:, 1] < image_shape[1] - half_patch_size
    far_enough_top = contour[:, 0] > half_patch_size
    far_enough_bottom = contour[:, 0] < image_shape[0] - half_patch_size
    full_patch_in_view = (
        far_enough_right & far_enough_left & far_enough_top & far_enough_bottom
    )
    return full_patch_in_view

def _extract_viewable_patches(points_in_view, image, contour, half_patch_size):
    viewable_contour = contour[points_in_view]
    patches = []
    for point in tqdm(viewable_contour):
        patch = image[
            floor(point[0] - half_patch_size) : ceil(point[0] + half_patch_size),
            floor(point[1] - half_patch_size) : ceil(point[1] + half_patch_size),
        ]
        patches.append(patch)
    patches = np.array(patches)
    return patches, viewable_contour

def extract_patches(
    image: np.ndarray, contour: np.ndarray, half_patch_size: int
) -> np.ndarray:
    """
    Extract patches from an image based on the contours
    """
    full_patch_in_view = _find_points_in_field_of_view(
        contour, half_patch_size, image.shape
    )
    patches, viewable_contour = _extract_viewable_patches(
        full_patch_in_view, image, contour, half_patch_size
    )

    return np.array(patches), viewable_contour

def calculate_growth_rate(
    image_stack: np.ndarray, half_patch_size: int, pre_growth_filter: Callable = None
):
    contours, filled_images = extract_long_contours_from_movie(image_stack)

    images_for_growth_calc = image_stack
    if pre_growth_filter is not None:
        images_for_growth_calc = pre_growth_filter(image_stack)

    growth_rates = []
    viewable_contours = []
    for i, contour in enumerate(contours[:-1]):
        full_patch_in_view = _find_points_in_field_of_view(
            contour, half_patch_size, image_stack[i].shape
        )
        image_i = images_for_growth_calc[i]
        image_i_plus_1 = images_for_growth_calc[i + 1]

        patches_at_time_t, viewable_contour  = _extract_viewable_patches(
            full_patch_in_view, image_i, contour, half_patch_size
        )
        patches_at_time_t_plus_1, _ = _extract_viewable_patches(
            full_patch_in_view, image_i_plus_1, contour, half_patch_size
        )

        dPatch = np.sum(patches_at_time_t_plus_1 - patches_at_time_t, axis=(1, 2))
        growth_rates.append(dPatch)
        viewable_contours.append(viewable_contour)
    
    return growth_rates, viewable_contours



