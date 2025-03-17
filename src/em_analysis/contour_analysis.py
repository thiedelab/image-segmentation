import numpy as np
from em_analysis import custom_feature as cf
from skimage import measure
from typing import List
from tqdm import tqdm


def extract_longest_contours_from_movie(raw_images: np.ndarray, round_to_pixel: bool=True) -> np.ndarray:
    """
    Extract the contours from an em movie
    """
    binarized_images = cf.binarize_image(raw_images)
    filled_images = cf.morphology.remove_small_holes(binarized_images).astype('float')
    contours = []
    for img_i in filled_images:
        contour_i = measure.find_contours(img_i, 0.5)
        contour_i = max(contour_i, key=len)
        if round_to_pixel:
            contour_i = (contour_i).astype('int')
        contours.append(contour_i)
    return contours, filled_images


def extract_long_contours_from_movie(raw_images: np.ndarray, min_length: int = 200, round_to_pixel: bool=True) -> np.ndarray:
    """
    Extract the contours from an em movie
    """
    binarized_images = cf.binarize_image(raw_images)
    filled_images = cf.morphology.remove_small_holes(binarized_images).astype('float')
    contours = []
    for img_i in filled_images:
        contour_i = measure.find_contours(img_i, 0.5)
        contour_i = np.vstack([c for c in contour_i if len(c) > min_length])
        if round_to_pixel:
            contour_i = (contour_i).astype('int')
        contours.append(contour_i)
    return contours, filled_images



def get_contours_of_length(contour, contour_length: int) -> List[np.ndarray]:
    contours_of_length = []
    for i, c_i in enumerate(contour):
        bwd_contour = []
        for j in range(i, 0, -1):
            c_ij = contour[j]
            if np.linalg.norm(c_i - c_ij) > contour_length:
                break
            bwd_contour.append(c_ij)
        bwd_contour = bwd_contour[::-1]

        fwd_contour = []
        for j, c_ij in enumerate(contour[i:]):
            if np.linalg.norm(c_i - c_ij) > contour_length:
                break
            fwd_contour.append(c_ij)
        contour_i = (
            bwd_contour
            + [
                c_i,
            ]
            + fwd_contour
        )
        contours_of_length.append(np.array(contour_i))
    return contours_of_length


def _get_major_minor_axis(contour):
    c_mean_zero = contour - np.mean(contour, axis=0)
    U, S, Vt = np.linalg.svd(c_mean_zero)
    return Vt, c_mean_zero


def _get_endpoint_rotation(contour):
    c_mean_zero = contour - np.mean(contour, axis=0)
    end_to_end_vec = c_mean_zero[-1] - c_mean_zero[0]
    end_to_end_vec /= np.linalg.norm(end_to_end_vec)
    ortho_vec = np.array([-end_to_end_vec[1], end_to_end_vec[0]])
    Vt = np.array([end_to_end_vec, ortho_vec])
    return Vt, c_mean_zero


def rotate_to_horizontal(contour, rotation="endpoints"):
    """
    Rotates a contour so that the major axis of contour is aligned horizontally
    """
    if rotation == "average":
        Vt, c_mean_zero = _get_major_minor_axis(contour)
    if rotation == "endpoints":
        Vt, c_mean_zero = _get_endpoint_rotation(contour)

    rotated_contour = c_mean_zero @ Vt.T

    return rotated_contour


def calculate_local_roughness(contour: np.ndarray, max_contour_length: float):
    contours_of_length = get_contours_of_length(contour, max_contour_length)

    rmsfs = []
    for c_i in contours_of_length:
        rotated_cotour = rotate_to_horizontal(c_i)
        rmsfs.append(np.std(rotated_cotour))
    return rmsfs, contours_of_length


def calculate_avg_roughness_per_time(raw_frames, contour_length=20):
    contours, filled_images = extract_longest_contours_from_movie(raw_frames)
    local_roughness = []
    for contour in tqdm(contours):
        rmsfs, contours_of_length = calculate_local_roughness(contour, contour_length)
        local_roughness.append(np.mean(rmsfs))
    return np.array(local_roughness)
