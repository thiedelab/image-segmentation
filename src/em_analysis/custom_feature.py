import math
from skimage import (
    filters,
    exposure,
    measure,
    morphology,
    # segmentation,
    # color,
    # graph,
    # io,
)
from skimage.segmentation import watershed

# from skimage.segmentation import random_walker
# from skimage.segmentation import active_contour
from skimage.feature import canny, peak_local_max
from scipy import linalg

# from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction import image as m_image
from scipy import ndimage as ndi
import numpy as np
from sklearn.cluster import spectral_clustering, DBSCAN
from scipy.optimize import curve_fit
import pims
import cv2
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import argparse


def load_images(data_path: str):
    # data_path = "data/f3 -0.1V.tif"

    # Open the image sequence
    raw_frames = pims.open(data_path)

    images = []
    for i in range(len(raw_frames)):
        image = raw_frames[i]
        image = np.array(image, dtype=np.float32)
        # Normalize
        image -= image.min()
        image /= image.max()
        # Crop for relevant features
        image = image[550:685, 340:895]
        # Append the cropped image
        images.append(image)
    return images


def binarize_image(image):
    smoothed_image = filters.gaussian(image, sigma=1)

    # thresholding using Otsu's method
    threshold_value = filters.threshold_otsu(smoothed_image)
    binary_image = smoothed_image >= threshold_value
    return binary_image


def segment_particles(image):
    # Apply Gaussian smoothing to reduce noise
    # Will may be change the area threshold
    binary_image = binarize_image(image)
    filled_image = morphology.remove_small_holes(binary_image, area_threshold=64)

    cleaned_image = morphology.remove_small_objects(filled_image, min_size=100)

    closed_image = morphology.binary_closing(cleaned_image, morphology.disk(3))

    return closed_image


def label_and_filter_regions(binary_image, min_size=900, max_size=9700):
    # Label connected components
    labeled_image, num_labels = measure.label(
        binary_image, return_num=True, connectivity=2
    )

    # Filter regions based on size
    filtered_regions = [
        region
        for region in measure.regionprops(labeled_image)
        if min_size < region.area < max_size
    ]

    return labeled_image, filtered_regions


# Probably a bad design to make a visualization function return a result of
# the visualizations. May be make it a separate function?
def visualize_results(image, regions):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    area_in_a_frame = []
    # Draw regions
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        get_image_embedded_in_rectangle(rect, image)
        ax.add_patch(rect)
        modified_image = get_image_embedded_in_rectangle(rect, image)
        area_in_a_frame.append(calculate_area_in_image(modified_image))

    plt.show()
    return area_in_a_frame


def enhance_contrast(image):
    enhanced_image = exposure.equalize_adapthist(image)
    return enhanced_image


def detect_edges(image):
    edges = canny(image, sigma=2)
    return edges


def segment_particles_with_edge_detection(image):
    edges = detect_edges(image)
    return edges


def get_image_embedded_in_rectangle(rectangle, image):
    # Get the corners of the rectangle
    nested_coordinates = rectangle.get_corners()
    bottom_left = nested_coordinates[0].astype(np.int64)
    bottom_right = nested_coordinates[1].astype(np.int64)
    top_right = nested_coordinates[2].astype(np.int64)
    top_left = nested_coordinates[3].astype(np.int64)
    # We now must extract the image
    modified_image = image[
        bottom_left[0] : bottom_right[0] + 1, bottom_right[1] : top_left[1] + 1
    ]
    return modified_image


def plot_contours(image, contours):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def edge_detector(image):
    contours = measure.find_contours(image, level=0.5)
    # plot_contours(image,contours)
    return contours


def image_separator(image, index, output_filename="sepatated.png"):
    # From Scipy
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=mask)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Overlapping objects")
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title("Distances")
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title("Separated objects")

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    plt.savefig("separated" + str(index) + ".png")
    plt.close()  # Close the figure to free up memory


def calculate_area_in_image(image):
    sum = np.sum(image)
    return sum


# Clustering based on the eigen vectors of the segmented image
def spectral_cluster_of_img(img, i):
    mask = img.astype(bool)
    graph = m_image.img_to_graph(img, mask=mask)
    graph.data = np.exp(-graph.data / (graph.data.std()))
    # We may have to modify the number of clusters
    labels = spectral_clustering(graph, n_clusters=4, eigen_solver="arpack")
    label_im = np.full(mask.shape, -1.0)
    label_im[mask] = labels

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].matshow(img)
    axs[1].matshow(label_im)

    plt.show()
    plt.savefig("spectral_cluster" + str(i) + ".png")
    plt.close()


def stack_contours(contour):
    return np.vstack(contour)


# def get_contour_patches(image,contour_point, delta = 10):
#     # padded_image = np.pad(image, pad_width=delta, mode='constant', constant_values=0)
#     # Since we padded the image by adding zeros, we must translate our coordinates

#     # Here we are getting overlapping patches. We don't want our patches to overlap
#     # We need to presave the previous so that we are not with in that boundary
#     v_dim, h_dim = image.shape
#     y = int(math.floor(contour_point[0]))
#     x = int(math.floor(contour_point[1]))

#     if x - delta >= 0 and y - delta >= 0 and x + delta < h_dim and y + delta < v_dim:
#         patch  = np.array(image[y - delta:y + delta + 1, x - delta:x + delta + 1])
#         return (patch,y,x)

#     # patch = padded_image[y - delta:y + delta + 1, x - delta:x + delta + 1]
#     # return patch


def get_contour_patches(image, contour_point, delta=10, prev_patch=None):
    # Get the image dimensions
    v_dim, h_dim = image.shape
    y = int(math.floor(contour_point[0]))
    x = int(math.floor(contour_point[1]))

    # Check if the patch fits within the image bounds
    if x - delta >= 0 and y - delta >= 0 and x + delta < h_dim and y + delta < v_dim:
        # Ensure that the new patch does not overlap with the previous one
        if prev_patch is not None:
            _, prev_y, prev_x = prev_patch
            # Check if the current patch overlaps with the previous patch
            if abs(y - prev_y) <= 2 * delta and abs(x - prev_x) <= 2 * delta:
                return None

        # Extract the patch from the image
        patch = np.array(image[y - delta : y + delta + 1, x - delta : x + delta + 1])

        # Return the patch along with its coordinates
        return (patch, y, x)


def plot_feature_matrix(feature_matrix, max_cols=100):
    # SCIPY + GPT
    for i, inner_row_matrix in enumerate(feature_matrix):
        n_patches = inner_row_matrix.shape[0]

        cols = min(max_cols, n_patches)
        rows = (n_patches + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(f"Feature Matrix {i}: {n_patches} Patches", fontsize=16)

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for j in range(n_patches):
            axes[j].imshow(inner_row_matrix[j], cmap="gray")
            axes[j].axis("off")
            axes[j].set_title(f"Patch {j+1}")

        for j in range(n_patches, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()


def get_feature_matrix(images):
    feature_matrix = []
    feature_matrix_with_pos = []
    ##### Instead of padding, we may have to get rid of points that are too close to the edge. Let's try to do that now.
    for i, image in enumerate(images):
        # Segment the image before we do anything else
        segmented_image = segment_particles(image)
        # Get the contours
        contour = edge_detector(segmented_image)
        # Get the length of the longest array
        max_length = len(max(contour, key=len))
        # Get the longest contour. There can only be 1, but we made it a trivial list of length 1 for a clean for loop
        longest_contour = [arr for arr in contour if len(arr) == max_length]
        # Set up our inner row matrix

        inner_row_matrix = []
        inner_row_matrix_pos = []
        prev_patch = None
        for points in longest_contour[0]:
            patch_result = get_contour_patches(
                image, points, delta=10, prev_patch=prev_patch
            )
            if patch_result is not None:
                patch, y, x = patch_result

                prev_patch = (patch, y, x)
                inner_row_matrix.append((patch))
                inner_row_matrix_pos.append({"patch": patch, "y_pos": y, "x_pos": x})

        inner_row_matrix = np.array(inner_row_matrix)
        inner_row_matrix_pos = np.array(inner_row_matrix_pos)
        feature_matrix.append(inner_row_matrix)
        feature_matrix_with_pos.append(inner_row_matrix_pos)

    return np.array(feature_matrix, dtype=object), np.array(
        feature_matrix_with_pos, dtype=object
    )


# Irrelevant now
def pad_patches(X, target_patches):
    current_patches = X.shape[0]
    if current_patches < target_patches:
        padding = np.zeros((target_patches - current_patches, X.shape[1]))
        X_padded = np.vstack([X, padding])
    else:
        X_padded = X
    return X_padded


# Revise the mathematical background for linear regression for a better analysis
def regress_features(feature_matrix):
    X, y = [], []
    for i in range(len(feature_matrix) - 1):
        current_row_matrix = feature_matrix[i]
        next_row_matrix = feature_matrix[i + 1]
        # (a,b,c) -> (a, b x c)

        flattened_X = current_row_matrix.reshape(current_row_matrix.shape[0], -1)
        flattened_y = next_row_matrix.reshape(next_row_matrix.shape[0], -1)
        target_patch = max(flattened_X.shape[0], flattened_y.shape[0])

        flattened_X = pad_patches(flattened_X, target_patch)
        flattened_y = pad_patches(flattened_y, target_patch)

        X.append(flattened_X)
        y.append(flattened_y)

    X = np.vstack(X)
    y = np.vstack(y)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    learned_matrix = model.coef_

    # print(f"Dimension of the learned matrix is: {learned_matrix.shape}")
    # mse = mean_squared_error(y,y_pred)
    # print(f"Mean squared error : {mse}")
    return learned_matrix


# Normalize the matrices to the range [0, 255] for image visualization
def normalize_to_image(matrix):
    norm_matrix = (matrix - matrix.min()) / (
        matrix.max() - matrix.min()
    )  # Normalize to [0, 1]
    return (norm_matrix * 255).astype(np.uint8)  # Scale to [0, 255]


def plot_singular_values_and_vectors(U, s, Vh):
    # Plot singular values
    top_4_left = [normalize_to_image(U[:, i].reshape(21, 21)) for i in range(4)]
    top_4_right = [normalize_to_image(Vh[i, :].reshape(21, 21)) for i in range(4)]

    plt.figure(figsize=(8, 6))
    plt.plot(s, marker="o", linestyle="-", markersize=5)
    plt.title("Singular Values of the Matrix", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Singular Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

    # Plot the top 4 left and right singular vectors
    for i in range(4):
        plt.figure(figsize=(10, 5))

        # Plot left singular vector
        plt.subplot(1, 2, 1)
        plt.imshow((top_4_left[i]), cmap="gray")
        plt.title(f"Left Singular Vector {i + 1}")
        plt.axis("off")

        # Plot right singular vector
        plt.subplot(1, 2, 2)
        plt.imshow((top_4_right[i]), cmap="gray")
        plt.title(f"Right Singular Vector {i + 1}")
        plt.axis("off")

        plt.suptitle(f"Singular Vector Pair {i + 1}")
        plt.tight_layout()
        plt.show()


# Try to read upon the mathematical background
def get_singular_value_decomp(learned_coeff):
    U, s, Vh = linalg.svd(learned_coeff)
    # U -> Unitary matrix having left singular vectors as columns
    # s -> singular values, sorted in non-increasing order of dim: min(M,N)
    # VH -> Unitary matrix having right singular vectors as rows

    # plot_singular_values_and_vectors(U,s,Vh)
    return U, s, Vh


def spanning_left_right_SV(U, Vh, dim=3):
    top_i_left = np.array([(U[:, i]) for i in range(dim)])
    top_i_right = np.array([(Vh[i, :]) for i in range(dim)])

    return top_i_left, top_i_right


# Define a degree 4 polynomial fitting function
def poly_func(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def plot_best_fit_curve(unitary_matrix):
    # Flatten each matrix and fit a curve
    for idx, matrix in enumerate(unitary_matrix):
        flattened = matrix.flatten()  # Flatten the matrix into 1D array
        x_data = np.arange(len(flattened))  # Create x values as indices
        y_data = flattened  # y values are the flattened matrix values

        # Fit the curve (degree 4 polynomial)
        popt, _ = curve_fit(poly_func, x_data, y_data)

        # Generate fitted curve
        fitted_curve = poly_func(x_data, *popt)

        # Plot the original data and fitted curve
        plt.figure(figsize=(8, 4))
        plt.scatter(x_data, y_data, label="Original Data", s=10)
        plt.plot(
            x_data,
            fitted_curve,
            label="Fitted Curve (Degree 4)",
            color="red",
            linewidth=2,
        )
        plt.title(f"Curve Fitting for Matrix {idx+1}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()


def dimensionality_reduction(data_points, spanning_set):
    projected_space = []
    print(f"Dimension of the spanning_set is: {spanning_set.shape}")
    for row_vector in data_points:
        reduced_space = row_vector @ spanning_set.T
        projected_space.append(np.array(reduced_space))
    # Having it as type object may be a problem we have to fix
    projected_space = np.array(projected_space, dtype=object)
    return projected_space


def visualize_dimensionally_reduced_feature(
    feature_matrix, left_sv_set, right_sv_set, feature_matrix_pos
):
    # We first reshape each entry of the feature matrix
    reshaped_feature = [
        feature.reshape(feature.shape[0], -1) for feature in feature_matrix
    ]
    # print(f"The shape of the first reshaped feature is: {reshaped_feature[0].shape}")
    reshaped_feature = np.array(reshaped_feature, dtype=object)

    left_dim_red_feature = dimensionality_reduction(reshaped_feature, left_sv_set)
    # print(f"The dimension of the first row vector of left_dim_red_feature is: {left_dim_red_feature[0].shape}")
    right_dim_red_feature = dimensionality_reduction(reshaped_feature, right_sv_set)

    left_dim_red_feature = np.array(left_dim_red_feature)
    right_dim_red_feature = np.array(right_dim_red_feature)
    # print(f"Dimension of left_red is: {left_dim_red_feature[0].shape}")
    # plot_3d_reduced_features(left_dim_red_feature)
    # plot_3d_reduced_features(right_dim_red_feature)

    # cluster_labels, clustered_images = cluster_and_trace_back_kmeans(left_dim_red_feature, feature_matrix)
    # visualize_clusters(left_dim_red_feature, cluster_labels)

    for idx, feature in enumerate(left_dim_red_feature):
        # This is a bit better I guess. NOw for each cluster point, we get the corresponding patches and we must indicate
        cluster_labels, clustered_images = cluster_and_trace_back_kmeans_single_feature(
            feature, feature_matrix_pos, idx
        )
        # visualize_clusters(feature, cluster_labels)
        # Why do we only see it for the first image
        visualize_clusters(feature, cluster_labels)
        print(f"The shape of the feature is: {feature.shape}")
        map_back_clusters_to_images(images, clustered_images, idx)

    # tot_num = 0
    # for cluster_label, patch_infos in clustered_images.items():
    #     # images = np.array(images)
    #     # tot_num += (len(images))
    #     # print(f"Num of images is: {images.shape}")
    #     # # mean_image = np.mean(images, axis = 0)
    #     # # print(f"Dimension of mean_image is: {mean_image.shape}")
    #     # for img in images:
    #     #     plt.figure(figsize=(3, 3))
    #     #     plt.imshow(segment_particles(img), cmap='gray')
    #     #     plt.title(f"Cluster {cluster_label} - Image {i+1}")
    #     #     plt.axis('off')
    #     #     plt.show()

    #     # images = np.array(images)
    #     # print(f"Num of images in Cluster {cluster_label} is: {images.shape}")

    #     # # Process images for creating a single full image grid
    #     # segmented_images = [segment_particles(img) for img in images]

    #     # # Calculate the grid size
    #     # num_images = len(segmented_images)
    #     # grid_cols =  int(np.ceil(np.sqrt(num_images)))

    #     # grid_rows = int(np.ceil(num_images / grid_cols))

    #     # # Create a canvas for the grid
    #     # img_height, img_width = segmented_images[0].shape
    #     # combined_image = np.zeros((grid_rows * img_height, grid_cols * img_width))

    #     # # Place each image in the grid
    #     # for idx, img in enumerate(segmented_images):
    #     #     row = idx // grid_cols
    #     #     col = idx % grid_cols
    #     #     combined_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = img

    #     combined_image = create_fixed_size_grid(images,135,555)
    #     # Display the combined image
    #     plt.figure(figsize=(15, 15))
    #     plt.imshow(combined_image, cmap='gray')
    #     plt.title(f"Cluster {cluster_label} - Combined Full Image")
    #     plt.axis('off')
    #     plt.show()
    return left_dim_red_feature, right_dim_red_feature


def map_back_clusters_to_images(
    original_images, clustered_images, index, patch_size=20, colormap="jet"
):
    # Create an empty list to store images with overlays
    cmap = cm.get_cmap(colormap, len(clustered_images))  # One color per cluster

    # Initialize the overlay image for the specified index
    overlaid_image = np.zeros((*original_images[index].shape, 3), dtype=np.float32)

    # Iterate through the clusters and add overlays for the specified index
    for cluster_label, patches in clustered_images.items():
        color = cmap(cluster_label)[:3]  # Get RGB color for the cluster (ignore alpha)

        for patch_info in patches:
            if patch_info["feature_idx"] == index:
                y, x = patch_info["y_pos"], patch_info["x_pos"]
                half_size = patch_size // 2

                y_start, y_end = max(0, y - half_size), min(
                    overlaid_image.shape[0], y + half_size
                )
                x_start, x_end = max(0, x - half_size), min(
                    overlaid_image.shape[1], x + half_size
                )

                overlaid_image[y_start:y_end, x_start:x_end] += np.array(color)

    # Clip values to ensure they are within [0, 1] range
    overlaid_image = np.clip(overlaid_image, 0, 1)

    # Visualize the overlaid image
    plt.figure(figsize=(10, 10))
    plt.imshow(segment_particles(original_images[index]), cmap="gray")
    plt.imshow(overlaid_image, alpha=0.5)  # Overlay with transparency
    plt.title(f"Image {index} with Cluster Overlay")
    plt.axis("off")
    plt.show()

    return overlaid_image


def cluster_and_trace_back_kmeans_single_feature(
    data, feature_matrix_pos, feature_idx, n_clusters=3, random_state=42
):
    flattened_data = np.array(data)
    # print(f"Dim of Flattened_data is: {flattened_data.shape}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(flattened_data)

    clustered_patches = {}

    for row_idx, label in enumerate(cluster_labels):
        patch_info = feature_matrix_pos[feature_idx][row_idx]

        if label not in clustered_patches:
            clustered_patches[label] = []

        clustered_patches[label].append(
            {
                "feature_idx": feature_idx,
                "patch": patch_info["patch"],
                "y_pos": patch_info["y_pos"],
                "x_pos": patch_info["x_pos"],
            }
        )

    return cluster_labels, clustered_patches


### Derived code.###
def create_fixed_size_grid(images, target_height, target_width):
    num_images = len(images)
    # Calculate the grid dimensions
    grid_cols = int(np.ceil(np.sqrt(num_images)))
    grid_rows = int(np.ceil(num_images / grid_cols))

    # Determine the size of each cell in the grid
    cell_height = target_height // grid_rows
    cell_width = target_width // grid_cols

    # Create a blank canvas for the grid
    combined_image = np.zeros((target_height, target_width), dtype=np.float32)

    # Place each image in the grid
    for idx, img in enumerate(images):
        # Resize each image to fit the cell size
        resized_img = cv2.resize(
            img, (cell_width, cell_height), interpolation=cv2.INTER_AREA
        )

        # Compute row and column positions in the grid
        row = idx // grid_cols
        col = idx % grid_cols

        # Compute the slice indices for placement
        start_row = row * cell_height
        end_row = start_row + cell_height
        start_col = col * cell_width
        end_col = start_col + cell_width

        # Place the resized image in the grid
        combined_image[start_row:end_row, start_col:end_col] = resized_img

    return combined_image


def cluster_and_trace_back_kmeans(
    data, feature_matrix, feature_matrix_pos, n_clusters=3, random_state=42
):
    # data here refers to the reduced feature
    flattened_data = np.vstack(data)
    print(f"Dim of Flattened_data is: {flattened_data.shape}")
    print(f"flattened_data is: {flattened_data}")
    # print(f"Dimension of feature_matrix is: {feature_matrix[0].shape}")
    # print(f"Dimension of flattened_data is: {flattened_data.shape}")
    total_points = [len(feature) for feature in data]
    cumulative_counts = np.cumsum([0] + total_points)

    # Fit k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(flattened_data)
    # How many clusters are there:
    print(f"The number of clusters: {len(cluster_labels)}")
    # We are considering all the images all at once
    # Reconstruct each row’s "origin" and group them by cluster label
    clustered_images = {}

    for idx, label in enumerate(cluster_labels):
        # Figure out which "feature block" (image/patch) this row comes from
        feature_idx = np.searchsorted(cumulative_counts, idx, side="right") - 1
        row_idx = idx - cumulative_counts[feature_idx]
        print(f"feature_idx, row_idx is: {(feature_idx, row_idx)}")
        patch_info = feature_matrix_pos[feature_idx][row_idx]

        # Populate dictionary of cluster -> rows
        if label not in clustered_images:
            clustered_images[label] = []

        # MAIN QUESTION: How can we get coordinate information so that we can draw
        # The question is, do we get any relevant coordinate information from the mapped_patch

        clustered_images[label].append(
            {
                "feature_idx": feature_idx,
                "patch": patch_info["patch"],
                "y_pos": patch_info["y_pos"],
                "x_pos": patch_info["x_pos"],
            }
        )

    return cluster_labels, clustered_images


def cluster_and_trace_back_DB_SCAN(data, feature_matrix, eps=0.17, min_samples=8):
    flattened_data = np.vstack(data)
    print(f"Dimension of flattened_data is: {flattened_data.shape}")
    total_points = [len(feature) for feature in data]  # Number of points per feature
    cumulative_counts = np.cumsum([0] + total_points)  # Track indices for mapping back

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(flattened_data)
    cluster_labels = clustering.labels_

    clustered_images = {}
    # I am not sure about this, especially about how the idx is chosen
    for idx, label in enumerate(cluster_labels):
        # if label == -1:
        #     continue

        feature_idx = np.searchsorted(cumulative_counts, idx, side="right") - 1
        row_idx = idx - cumulative_counts[feature_idx]
        # From the flattened(reduced) feature we reconstruct the get the original patches.
        # Let's do a visual comparision between the image we get from mapping back and the actual images in the feature matrix.

        if label not in clustered_images:
            clustered_images[label] = []
        # if(feature_idx <= 10):
        #     print(f"(feature_idx, idx, cumulative_counts[feature_idx]): {(feature_idx,idx, cumulative_counts[feature_idx])}")
        clustered_images[label].append(feature_matrix[feature_idx][row_idx])

    return cluster_labels, clustered_images


def visualize_clusters(data, cluster_labels, color_map="jet"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    cmap = cm.get_cmap(color_map, max(cluster_labels) + 1)
    # Coloring mechanism
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label == -1:
            color = "k"  # Black for noise
            label_name = "Noise"
        else:
            color = cmap(label)[:3]  # Get the RGB color for the cluster
            label_name = f"Cluster {label}"

        cluster_points = np.vstack(data)[cluster_labels == label]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            label=label_name,
            color=color,
            s=10,
        )

    ax.set_title("3D Clustering Visualization")
    ax.set_xlabel("Φ_1")
    ax.set_ylabel("Φ_2")
    ax.set_zlabel("Φ_3")
    plt.legend()
    plt.show()


def plot_3d_reduced_features(reduced_features):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("Φ_1")
    ax.set_ylabel("Φ_2")
    ax.set_zlabel("Φ_3")
    for feature in reduced_features:
        # The first feature for example is a feature with dim 1221 x 3
        ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2])
    plt.show()


def plot_feature_images(feature_matrix, images):
    # After plotting each cluster image, we now plot the aggregate of each row in the feature matrix, to get an idea of what it is supposed to look like
    # Manually check the first patch and look at it side by side

    for idx, feature in enumerate(feature_matrix):
        patches = feature[:20]

        for i, (patch, y_pos, x_pos) in enumerate(patches):
            fig, axes = plt.subplots(
                1, 4, figsize=(30, 30)
            )  # Create a figure with 1 row and 3 columns
            # Plot the patch

            # It could be the case that there is an issue with how the patches are generated
            axes[0].imshow(segment_particles(patch))
            axes[0].set_title(f"{i}th patch centered at {(x_pos, y_pos)}")
            axes[0].axis("off")

            # Plot the full image with zooming based on patch center
            axes[1].imshow(segment_particles(images[idx]))
            axes[1].set_title(f"{idx}th image zoomed in")

            # Dynamically adjust the zoom region based on the current patch's coordinates
            zoom_size = (
                20  # Adjust based on the size of the region you want to zoom in on
            )
            axes[1].set_xlim(x_pos - zoom_size, x_pos + zoom_size)
            axes[1].set_ylim(y_pos - zoom_size, y_pos + zoom_size)
            axes[1].axis("off")

            manual_patch = images[idx][y_pos - 10 : y_pos + 11, x_pos - 10 : x_pos + 11]
            axes[2].imshow(segment_particles(manual_patch))
            axes[2].set_title(f"Actual image patch based on the center ")
            axes[2].axis("off")

            axes[3].imshow(segment_particles(images[idx]))
            axes[3].set_title("Original image")
            axes[3].axis("off")

            fig.suptitle(f"Comparison of {i}th patch and {idx}th image", fontsize=14)
            # May be we should add positional encoding as well
            # Show the figure
            plt.show()


# There is something wrong with how we generate the patches. We may have to fix something.


def pad_feature_matrices(feature_matrix):
    """Pads all matrices to have the same number of rows by adding zero matrices."""

    max_rows = max(
        mat.shape[0] for mat in feature_matrix
    )  # Find the max number of rows
    print(f"Max rows: {max_rows}")

    padded_matrices = []

    for mat in feature_matrix:
        current_rows = mat.shape[0]
        print(f"Before padding, rows: {current_rows}")

        # If there are missing rows, pad with zero (21x21) matrices
        if current_rows < max_rows:
            diff = max_rows - current_rows
            zero_pad = np.zeros((diff, 21, 21))  # Create zero-padding
            padded_mat = np.concatenate((mat, zero_pad), axis=0)  # Append padding
        else:
            padded_mat = mat  # No padding needed

        print(f"After padding, rows: {padded_mat.shape[0]}")
        padded_matrices.append(padded_mat)

    return np.array(padded_matrices)  # Convert list to a NumPy array


def rate_of_growth(feature_matrix):
    """Computes the rate of growth by finding the difference between consecutive frames."""
    print(f"Shape of feature matrix before padding: {np.shape(feature_matrix)}")

    feature_matrix = pad_feature_matrices(feature_matrix)  # Ensure consistent shape
    print(f"Shape of feature matrix after padding: {np.shape(feature_matrix)}")

    growth = [np.zeros_like(feature_matrix[0])]  # Initialize with a zero matrix

    # Compute the growth as the difference between consecutive frames
    for i in range(1, len(feature_matrix)):
        growth.append(feature_matrix[i] - feature_matrix[i - 1])

    return np.array(growth)  # Convert list back to NumPy array


def plot_growth_per_column_subplots(growth):
    num_timesteps = len(growth)
    num_rows, num_cols, _ = growth[0].shape

    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(15, 5), sharey=True)

    for col in range(num_cols):
        ax = axes[col] if num_cols > 1 else axes

        # Loop through each row and extract its values over time
        for row in range(num_rows):
            patch_values = [
                growth[t][row, col] for t in range(num_timesteps)
            ]  # Extract growth for (row, col)
            ax.plot(
                range(num_timesteps),
                patch_values,
                marker="o",
                linestyle="-",
                label=f"Row {row}",
            )

        ax.set_xlabel("Time")
        ax.set_title(f"Column {col}")
        ax.grid(True)
        if col == 0:
            ax.set_ylabel("Growth Value")
        ax.legend(fontsize="small", loc="upper right")

    plt.tight_layout()  # Adjust layout to avoid overlapping
    plt.show()


def plot_growth_graph(growth, patch_index):
    num_timesteps = len(growth)
    num_rows, num_cols = growth[0].shape  # Assuming all matrices have the same shape

    if patch_index >= num_cols:
        print(f"Invalid patch index {patch_index}. Max index is {num_cols - 1}.")
        return

    plt.figure(figsize=(12, 6))

    # Track the growth change for the selected patch across all timesteps
    growth_values = []
    for t in range(num_timesteps):
        # Gather the values for the specific patch (column) for the current timestep (row-wise)
        growth_values.extend(growth[t][:, patch_index])

    # Plotting
    plt.plot(range(len(growth_values)), growth_values, label=f"Patch {patch_index}")
    plt.title(f"Growth Values Over Time for Patch {patch_index}")
    plt.xlabel("Time")
    plt.ylabel("Growth Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def plot_growth_changes(feature_matrix, patch_index):
    num_timesteps = len(feature_matrix)
    patch_height, patch_width = (
        feature_matrix[0].shape[1],
        feature_matrix[0].shape[2],
    )  # 21x21 size

    growth_values = []

    # Track the sum of changes (or mean/max) for each patch over time
    for t in range(1, num_timesteps):
        # Calculate the difference between consecutive timesteps for the selected patch
        growth = np.abs(
            feature_matrix[t][patch_index] - feature_matrix[t - 1][patch_index]
        )
        growth_sum = np.sum(
            growth
        )  # Sum of pixel-wise changes (you can replace this with mean/max if desired)
        growth_values.append(growth_sum)

    # Plotting the quantified growth (sum of changes)
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(1, num_timesteps), growth_values, label=f"Patch {patch_index} Growth"
    )
    plt.title(f"Change in Growth Over Time for Patch {patch_index}")
    plt.xlabel("Time (Timestep)")
    plt.ylabel("Total Growth (Sum of Pixel Changes)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def calculate_average_growth_rate_bin(feature_matrix):
    num_timesteps = len(feature_matrix)
    # We dont need to pad here. We don't care about the patch growth yet

    # Track the sum of changes for the entire material over time
    growth_rates = []
    for t in range(1, num_timesteps):
        # Calculate the difference between consecutive timesteps for all patches in the material
        diff = sum_patches(feature_matrix[t]) - sum_patches(feature_matrix[t - 1])
        print(f"diff is {diff}")
        # Total sum of the white pixels
        growth_rates.append(diff)

    timesteps = np.arange(1, feature_matrix.shape[0])

    # Distribution of growth rate over time (Bar plot instead of Histogram)
    plt.subplot(2, 1, 1)  # First subplot
    plt.bar(timesteps, growth_rates, color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Growth rate")
    plt.title("Avergae Binarized Growth rate")

    # Line plot for growth rate over time
    plt.subplot(2, 1, 2)  # Second subplot
    plt.plot(
        timesteps,
        growth_rates,
        marker="o",
        linestyle="-",
        color="blue",
        label="Growth Rate",
    )
    plt.xlabel("Timestep")
    plt.ylabel("Growth Rate")
    plt.title("Average Binarized Growth Rate Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# A small helper to sum over entire patches. We may require some overlapping behaviour between the patches.
def sum_patches(patches):
    accum = np.zeros((21, 21))
    for patch in patches:
        accum += patch
    return np.sum(accum)


# Avergage growth for the original set of images(NON-BINARIZED)
def org_avergage_growth_rate(images):
    rate_change = []
    for i in range(1, len(images)):
        rate_change.append(np.sum(image[i] - image[i - 1]))
    x_axis = np.arange(1, len(rate_change) + 1)

    plt.figure(figsize=(10, 6))

    # Distribution of growth rate over time (Bar plot instead of Histogram)
    plt.subplot(2, 1, 1)  # First subplot
    plt.bar(x_axis, rate_change, color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Original Growth rate")
    plt.title("Avergae Original Growth rate")

    # Line plot for growth rate over time
    plt.subplot(2, 1, 2)  # Second subplot
    plt.plot(
        x_axis,
        rate_change,
        marker="o",
        linestyle="-",
        color="blue",
        label="Growth Rate",
    )
    plt.xlabel("Timestep")
    plt.ylabel("Peak Growth Rate")
    plt.title("Average Original Growth Rate Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Global maxima
def plot_height_growth_rate(images):
    peak_growth_rate = []
    # It would be cool if we can definitively determine whether whatever structure we are on is dendritic or not
    for i, image in enumerate(images):
        # It is advised that we segment here
        segmented_image = segment_particles(image)
        contour = edge_detector(segmented_image)
        # Get the length of the longest array
        # print(max(contour, key=len))
        # What exactly does the contour look like

        max_length = len(max(contour, key=len))
        # Get the longest contour. There can only be 1, but we made it a trivial list of length 1 for a clean "for loop"
        longest_contour = [arr for arr in contour if len(arr) == max_length]
        # We need the maximum y value

        peak_height = np.max(longest_contour[0][:, 1])
        # print(f"Peak_height is: {peak_height}")
        peak_growth_rate.append(peak_height)
    # We plot the change now
    x_axis = np.arange(1, 61)
    plt.figure(figsize=(10, 8))
    plt.plot(x_axis, peak_growth_rate)
    plt.xlabel("Timestep")
    plt.ylabel("Average Growth Rate")
    plt.title(
        "Signed Maximum Height Growth Rate Over Time for Entire Binarized Material"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_intermediate_steps(
    original_image,
    preprocessed_image,
    enhanced_image,
    segmented_image,
    edges_image,
    final_image,
    regions,
):
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    # Plot the different images
    ax[0, 0].imshow(original_image, cmap="gray")
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(preprocessed_image, cmap="gray")
    ax[0, 1].set_title("Preprocessed Image")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(enhanced_image, cmap="gray")
    ax[1, 0].set_title("Enhanced Image")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(segmented_image, cmap="gray")
    ax[1, 1].set_title("Segmented Image")
    ax[1, 1].axis("off")

    ax[2, 0].imshow(edges_image, cmap="gray")
    ax[2, 0].set_title("Edges Image")
    ax[2, 0].axis("off")

    ax[2, 1].imshow(original_image, cmap="gray")
    ax[2, 1].set_title("Final Results")
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax[2, 1].add_patch(rect)
        print("The area of t")
    ax[2, 1].axis("off")

    plt.tight_layout()
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(description="Custom Feature Analysis")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/f3 -0.1V.tif",
        help="Path to the data file",
    )
    return parser.parse_args()


def main(args):
    data_path = args.data_path
    # data_path = "data/f3 -0.1V.tif"
    images = load_images(data_path)
    org_avergage_growth_rate(images)
    plot_height_growth_rate(images)
    feature_matrix, feature_matrix_pos = get_feature_matrix(images)
    calculate_average_growth_rate_bin(feature_matrix)

    # growth = rate_of_growth(feature_matrix)
    # curr = 0
    # #
    # plot_average_growth_rate(feature_matrix)
    # for i in range(len(feature_matrix[0])):

    #     plot_growth_changes(feature_matrix,i)
    #     calculate_average_growth_rate(feature_matrix,i)
    # plot_growth_per_column_subplots(growth)
    # print(f"Dim of feature_matrix: {feature_matrix}")
    # plot_feature_images(feature_matrix,images)

    learned_coeff = regress_features(feature_matrix)
    # U,_,Vh = get_singular_value_decomp(learned_coeff)
    # left_spanning, right_spanning = spanning_left_right_SV(U,Vh)
    # left, _ = visualize_dimensionally_reduced_feature(feature_matrix,left_spanning,right_spanning,feature_matrix_pos)

    # Example usage:
    for i, image in enumerate(images):
        # # Step 1: Apply manual thresholding - easier to experiment with
        # There is an issue with our thresholding
        # enhanced_image = image >= 0.41

        # Step 2: Use edge detection for segmentation
        segmented_image = segment_particles(image)

        # Step 3: Label and filter regions
        labeled_image, regions = label_and_filter_regions(segmented_image)
        # Step 4: Visualize the results

        visualize_results(segmented_image, regions)
        edge_file_name = "detected_edges" + str(i) + ".png"
        # plot_contours(labeled_image, contours)
        # edge_detector(labeled_image, output_filename = edge_file_name)


if __name__ == "__main__":
    args = _parse_args()
    main()
