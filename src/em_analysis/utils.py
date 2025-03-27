import numpy as np


def scale_images(image_stack: np.ndarray) -> np.ndarray:
    """
    Scales a stack of images to the range [0, 1] using the 5th and 95th percentiles.

    Parameters
    ----------
    image_stack : np.ndarray
        A 3D numpy array representing a stack of images. The shape should be (n_images, height, width).

    Returns
    -------
    np.ndarray
        A 3D numpy array with the same shape as input, where each image is scaled to the range [0, 1].
        The scaling is done using the 5th and 95th percentiles of the pixel values in the stack.
    """
    image_stack = image_stack.astype(np.float32)
    # image_mins = np.min(image_stack, axis=(1, 2), keepdims=True

    p_low, p_high = np.percentile(image_stack, [5, 95])
    clipped = image_stack - p_low
    clipped[clipped < 0.0] = 0.0
    clipped = clipped / p_high
    clipped[clipped > 1.0] = 1.0
    return clipped
