import numpy as np
from em_analysis import custom_feature as cf
from skimage import measure, filters
from typing import List, Callable


def calculate_growth_rate_at_contour(
    video,
    pixel_size_in_nm,
    binarization_sigma=2.0,
    min_length=200,
    pre_filter=None,
    threshold_function=None,
):
    """
    Calculates the local growth rate along a contour in a video.

    Parameters
    ----------
    video: np.ndarray
        A 3D numpy array representing the video. The shape should be (n_frames, height, width).
    pixel_size_in_nm: np.ndarray
        A 1D numpy array representing the pixel size in nanometers for each frame.
    binarization_sigma: float
        The standard deviation for Gaussian kernel used as a blur prior to binarization.
    min_length: int
        The minimum length of contours to be considered.
    pre_filter: Callable
        A function to apply a pre-filter to the video before processing.
    threshold_function: Callable
        A function to apply a threshold to the video. If None, a multi-Otsu threshold is used
        and the lowest class is returned.

    Returns
    -------
    raw_contours: List[np.ndarray]
        A list of contours extracted from the video.
    growth_rate_at_contour: List[np.ndarray]
        The value of the local growth rate estimated at each point in the contour
    growth_rate_video: np.ndarray
        The estimated growth rate everywhere in the image.

    TODO
    ----
    It might be cleaner to pass a single binarization function as an argument rather than
    passing the sigma and threshold function separately?
    """
    if pre_filter is not None:
        video = pre_filter(video)

    if threshold_function is None:

        def threshold_function(x):
            thr = filters.threshold_multiotsu(x, classes=4)[0]
            return x > thr

    binarize_function = lambda x: cf.binarize_image(
        x, sigma=binarization_sigma, threshold_function=threshold_function
    )
    raw_contours, __ = extract_long_contours_from_movie(
        video, min_length=min_length, binarization_function=binarize_function
    )

    growth_rate_filter = lambda x: filters.gaussian(
        x, sigma=np.array([5, 5]), mode="reflect"
    )
    filtered_at_time = np.array([growth_rate_filter(x) for x in video])
    growth_rate_video = np.zeros_like(filtered_at_time)

    # Central difference for middle points
    growth_rate_video[1:-1] = (filtered_at_time[2:] - filtered_at_time[:-2]) / (
        2.0 * pixel_size_in_nm
    )
    # Forward difference for first point
    growth_rate_video[0] = (
        filtered_at_time[1] - filtered_at_time[0]
    ) / pixel_size_in_nm
    # Backward difference for last point
    growth_rate_video[-1] = (
        filtered_at_time[-1] - filtered_at_time[-2]
    ) / pixel_size_in_nm

    growth_rate_at_contour = _query_video_at_contour(raw_contours, growth_rate_video)

    return raw_contours, growth_rate_at_contour, growth_rate_video


def extract_n_longest_contours_from_movie(
    raw_images: np.ndarray,
    n_longest: int = 1,
    round_to_pixel: bool = True,
    binarization_function: Callable = None,
) -> np.ndarray:
    """
    Extract the contours from an em movie
    """
    if binarization_function is None:
        binarization_function = cf.binarize_image
    binarized_images = binarization_function(raw_images)
    filled_images = cf.morphology.remove_small_holes(binarized_images).astype("float")
    contours = []
    for img_i in filled_images:
        contour_i = measure.find_contours(img_i, 0.5)
        # Sort contours by length
        contour_i = sorted(contour_i, key=lambda x: len(x), reverse=True)
        contour_i = np.vstack(contour_i[:n_longest])
        if round_to_pixel:
            contour_i = (contour_i).astype("int")
        contours.append(contour_i)
    return contours, filled_images


def extract_long_contours_from_movie(
    raw_images: np.ndarray,
    min_length: int = 200,
    round_to_pixel: bool = True,
    binarization_function: Callable = None,
) -> np.ndarray:
    """
    Extract the contours from an em movie
    """
    if binarization_function is None:
        binarization_function = cf.binarize_image
    binarized_images = binarization_function(raw_images)
    filled_images = cf.morphology.remove_small_holes(binarized_images).astype("float")
    contours = []
    for img_i in filled_images:
        contour_i = measure.find_contours(img_i, 0.5)
        contour_i = np.vstack([c for c in contour_i if len(c) > min_length])
        if round_to_pixel:
            contour_i = (contour_i).astype("int")
        contours.append(contour_i)
    return contours, filled_images


def _query_video_at_contour(raw_contours, video):
    growth_rate_at_contour = []

    for i, contour in enumerate(raw_contours):
        current_increase = video[i]
        contour_points = current_increase[contour[:, 0], contour[:, 1]]
        growth_rate_at_contour.append(contour_points)
    return growth_rate_at_contour


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
    for contour in contours:
        rmsfs, contours_of_length = calculate_local_roughness(contour, contour_length)
        local_roughness.append(np.mean(rmsfs))
    return np.array(local_roughness)


def calculate_intensity_change_images(
    video, spatial_sigma: float = 5, pixel_size_in_nm: float = 1
):
    """
    Creates a video of changes in intensity between frames.  Should probably be refactored into a different file.
    """
    growth_rate_filter = lambda x: filters.gaussian(
        x, sigma=np.array([spatial_sigma, spatial_sigma]), mode="reflect"
    )
    filtered_at_time = [growth_rate_filter(x) for x in video]

    # Calculate the rate of increase using the central difference method
    rate_increase = (
        np.array(filtered_at_time[2:]) - np.array(filtered_at_time[:-2])
    ) / (2.0 * pixel_size_in_nm)
    return rate_increase


def calculate_contour_intensity_growth(
    video, contour_sigma, threshold_function, min_length=200
):
    binarize_function = lambda x: cf.binarize_image(
        x, sigma=contour_sigma, threshold_function=threshold_function
    )
    raw_contours, __ = extract_long_contours_from_movie(
        video, min_length=min_length, binarization_function=binarize_function
    )
    difference_images = calculate_intensity_change_images(video)

    rate_increase_at_contour = []
    contours_with_rates = []
    for i, contour in enumerate(raw_contours[1:-1]):
        current_difference = difference_images[i]
        contour_points = current_difference[contour[:, 0], contour[:, 1]]
        rate_increase_at_contour.append(contour_points)
        contours_with_rates.append(contour)
    return rate_increase_at_contour, contours_with_rates
