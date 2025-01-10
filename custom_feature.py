from skimage import filters, exposure, measure, morphology,segmentation,color,graph, io
from skimage.segmentation import watershed, random_walker, active_contour
from skimage.feature import canny, peak_local_max
from scipy import linalg
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction import image as m_image
from scipy import ndimage as ndi
import numpy as np
from sklearn.cluster import spectral_clustering, DBSCAN
from scipy.optimize import curve_fit
import pims
from collections import Counter

data_path = "data/f3 -0.1V.tif"

# Open the image sequence
raw_frames = pims.open(data_path)


images = []
for i in range(len(raw_frames)):
    image = raw_frames[i]
    image = np.array(image, dtype=np.float32)
    # Normalize
    image -= image.min() 
    image /= image.max()
    #Crop for relevant features
    image = image[550:685, 340:895]
    # Append the cropped image
    images.append(image)

def segment_particles(image):
    # Apply Gaussian smoothing to reduce noise
    smoothed_image = filters.gaussian(image, sigma=1)
    
    # thresholding using Otsu's method
    threshold_value = filters.threshold_otsu(smoothed_image)
    
    binary_image = smoothed_image >= threshold_value
    # Will may be change the area threshold
    filled_image = morphology.remove_small_holes(binary_image, area_threshold=64)
    
    cleaned_image = morphology.remove_small_objects(filled_image, min_size=100)
 
    closed_image = morphology.binary_closing(cleaned_image, morphology.disk(3))
    
    return closed_image

def label_and_filter_regions(binary_image, min_size=900, max_size=9700):
    # Label connected components
    labeled_image, num_labels = measure.label(binary_image, return_num=True, connectivity=2)
    
    # Filter regions based on size
    filtered_regions = [region for region in measure.regionprops(labeled_image)
                        if min_size < region.area < max_size]
    
    return labeled_image, filtered_regions


# Probably a bad design to make a visualization function return a result of the visualizations. May be make it a separate function?
def visualize_results(image, regions):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    area_in_a_frame = []
    # Draw regions
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        get_image_embedded_in_rectangle(rect,image)
        ax.add_patch(rect)
        modified_image = get_image_embedded_in_rectangle(rect,image)
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

def get_image_embedded_in_rectangle(rectangle,image):
    #Get the corners of the rectangle
    nested_coordinates = rectangle.get_corners()
    bottom_left = nested_coordinates[0].astype(np.int64)
    bottom_right = nested_coordinates[1].astype(np.int64)
    top_right = nested_coordinates[2].astype(np.int64)
    top_left = nested_coordinates[3].astype(np.int64)
    #We now must extract the image 
    modified_image = image[bottom_left[0] : bottom_right[0] + 1, bottom_right[1] : top_left[1] + 1]
    return modified_image

def plot_contours(image,contours):
    fig, ax = plt.subplots()
    ax.imshow(image,cmap="gray")
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()

def edge_detector(image, output_filename='detected_edges.png'):
    contours = measure.find_contours(image, level=0.5)
    #plot_contours(image,contours)
    return contours
    
def image_separator(image, index,output_filename = "sepatated.png"):
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
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    plt.savefig("separated" + str(index) + ".png")
    plt.close()  # Close the figure to free up memory   

def calculate_area_in_image(image):
    sum = np.sum(image)
    return sum

## Clustering based on the eigen vectors of the segmented image
def spectral_cluster_of_img(img,i):
    mask = img.astype(bool)
    graph = m_image.img_to_graph(img, mask = mask)
    graph.data = np.exp(-graph.data/(graph.data.std()))
    # We may have to modify the number of clusters
    labels = spectral_clustering(graph, n_clusters = 4, eigen_solver = "arpack")
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

def get_contour_patches(image,contour_point, delta = 10):
    # padded_image = np.pad(image, pad_width=delta, mode='constant', constant_values=0)
    # Since we padded the image by adding zeros, we must translate our coordinates
    v_dim, h_dim = image.shape
    y = int(round(contour_point[0]))
    x = int(round(contour_point[1]))
    if x - delta >= 0 and y - delta >= 0 and x + delta < h_dim and y + delta < v_dim:
        patch  = np.array(image[y - delta:y + delta + 1, x - delta:x + delta + 1])
        return patch
    
    # patch = padded_image[y - delta:y + delta + 1, x - delta:x + delta + 1]
    # return patch

def plot_feature_matrix(feature_matrix, max_cols = 100):
    # SCIPY + GPT
     for i, inner_row_matrix in enumerate(feature_matrix):
     
        n_patches = inner_row_matrix.shape[0] 
        
        cols = min(max_cols, n_patches)
        rows = (n_patches + cols - 1) // cols  

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(f"Feature Matrix {i}: {n_patches} Patches", fontsize=16)

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for j in range(n_patches):
            axes[j].imshow(inner_row_matrix[j], cmap='gray')
            axes[j].axis('off')
            axes[j].set_title(f"Patch {j+1}")
            
        for j in range(n_patches, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
        
def get_feature_matrix(images):
    feature_matrix = []
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
        for points in longest_contour[0]:
            patch = get_contour_patches(image,points)
            # Just in case
            if patch is not None:
                # print(f"Shape of patch is: {patch.shape}")
                ### Added the coordinates of the contours for visualization purposes. Must revert back.
                # inner_row_matrix.append((patch, points[0], points[1]))
                inner_row_matrix.append(patch)
        inner_row_matrix = np.array(inner_row_matrix)
        feature_matrix.append(inner_row_matrix)
    return np.array(feature_matrix, dtype = object)

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
    X,y = [],[]
    for i in range(len(feature_matrix) - 1):

        current_row_matrix = feature_matrix[i]
        next_row_matrix = feature_matrix[i + 1]
        #(a,b,c) -> (a, b x c)
        
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
    model.fit(X,y)
    
    y_pred = model.predict(X)
    
    learned_matrix = model.coef_ 
    
    #print(f"Dimension of the learned matrix is: {learned_matrix.shape}")
    # mse = mean_squared_error(y,y_pred)
    # print(f"Mean squared error : {mse}")
    return learned_matrix

# Normalize the matrices to the range [0, 255] for image visualization
def normalize_to_image(matrix):
    norm_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())  # Normalize to [0, 1]
    return (norm_matrix * 255).astype(np.uint8)  # Scale to [0, 255]

def plot_singular_values_and_vectors(U,s, Vh):
    # Plot singular values
    top_4_left = [normalize_to_image(U[:,i].reshape(21, 21)) for i in range(4)]
    top_4_right = [normalize_to_image(Vh[i,:].reshape(21, 21)) for i in range(4)]
    
    plt.figure(figsize=(8, 6))
    plt.plot(s, marker='o', linestyle='-', markersize=5)
    plt.title("Singular Values of the Matrix", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Singular Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
   
    # Plot the top 4 left and right singular vectors
    for i in range(4):
        plt.figure(figsize=(10, 5))
        
        # Plot left singular vector
        plt.subplot(1, 2, 1)
        plt.imshow((top_4_left[i]), cmap='gray')
        plt.title(f'Left Singular Vector {i + 1}')
        plt.axis('off')
        
        # Plot right singular vector
        plt.subplot(1, 2, 2)
        plt.imshow((top_4_right[i]), cmap='gray')
        plt.title(f'Right Singular Vector {i + 1}')
        plt.axis('off')
        
        plt.suptitle(f'Singular Vector Pair {i + 1}')
        plt.tight_layout()
        plt.show()

#Try to read upon the mathematical background
def get_singular_value_decomp(learned_coeff):
    U, s, Vh = linalg.svd(learned_coeff)
    # U -> Unitary matrix having left singular vectors as columns
    # s -> singular values, sorted in non-increasing order of dim: min(M,N)
    # VH -> Unitary matrix having right singular vectors as rows
    
    #plot_singular_values_and_vectors(U,s,Vh)
    return U,s,Vh

def spanning_left_right_SV(U, Vh, dim = 3):
    
    top_i_left = np.array([(U[:,i]) for i in range(dim)])
    top_i_right = np.array([(Vh[i,:]) for i in range(dim)])
    
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
        plt.plot(x_data, fitted_curve, label="Fitted Curve (Degree 4)", color="red", linewidth=2)
        plt.title(f"Curve Fitting for Matrix {idx+1}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

def dimensionality_reduction(data_points, spanning_set):
    projected_space = []
    for row_vector in data_points:
        
        reduced_space = row_vector @ spanning_set.T
        projected_space.append(np.array(reduced_space)) 
        
    projected_space = np.array(projected_space, dtype=object)     
    return projected_space

def visualize_dimensionally_reduced_feature(feature_matrix, left_sv_set, right_sv_set):
    # We first reshape each entry of the feature matrix
    reshaped_feature = [feature.reshape(feature.shape[0],-1) for feature in feature_matrix]
    
    reshaped_feature = np.array(reshaped_feature, dtype=object)
 
    left_dim_red_feature = dimensionality_reduction(reshaped_feature,left_sv_set)
    right_dim_red_feature = dimensionality_reduction(reshaped_feature,right_sv_set)
    
    left_dim_red_feature = np.array(left_dim_red_feature)    
    right_dim_red_feature = np.array(right_dim_red_feature)
    # print(f"Dimension of left_red is: {left_dim_red_feature.shape}")
    # plot_3d_reduced_features(left_dim_red_feature)
    # plot_3d_reduced_features(right_dim_red_feature)
    
    cluster_labels, clustered_images = cluster_and_trace_back(left_dim_red_feature, feature_matrix)
    visualize_clusters(left_dim_red_feature, cluster_labels)
    
    for cluster_label, images in clustered_images.items():
        images = np.array(images)
        print(f"Num of images is: {len(images)}")
        # mean_image = np.mean(images, axis = 0)
        # print(f"Dimension of mean_image is: {mean_image.shape}")
        for img in images:
            plt.figure(figsize=(3, 3))
            plt.imshow(segment_particles(img), cmap='gray')  
            plt.title(f"Cluster {cluster_label} - Image {i+1}")
            plt.axis('off')
            plt.show()
    
    

def cluster_and_trace_back(data, feature_matrix, eps=0.17, min_samples=7):
    flattened_data = np.vstack(data) 
    print(f"Dimension of flattened_data is: {flattened_data.shape}")
    total_points = [len(feature) for feature in data]  # Number of points per feature
    cumulative_counts = np.cumsum([0] + total_points)  # Track indices for mapping back
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(flattened_data)
    cluster_labels = clustering.labels_
    # What is the point of the flattened feature then? -> To analyze the cluster.
   
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

def visualize_clusters(data, cluster_labels):
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')

    # Coloring mechanism
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label == -1:
            color = "k"  # Black for noise
            label_name = "Noise"
        else:
            color = plt.cm.get_cmap("tab20")(label / (max(unique_labels) + 1))
            label_name = f"Cluster {label}"

        cluster_points = np.vstack(data)[cluster_labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=label_name, color=color, s=10)

    ax.set_title("3D Clustering Visualization")
    ax.set_xlabel("Φ_1")
    ax.set_ylabel("Φ_2")
    ax.set_zlabel("Φ_3")
    plt.legend()
    plt.show()


def plot_3d_reduced_features(reduced_features):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Φ_1')
    ax.set_ylabel('Φ_2')
    ax.set_zlabel('Φ_3')
    for feature in reduced_features:
         # The first feature for example is a feature with dim 1221 x 3
        ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2])
    plt.show()    

def plot_feature_images(feature_matrix, images):
    # After plotting each cluster image, we now plot the aggregate of each row in the feature matrix, to get an idea of what it is supposed to look like
    for idx, feature in enumerate(feature_matrix):
        patches = feature[:40]
        for i, (patch, y_pos, x_pos) in enumerate(patches):
            fig, axes = plt.subplots(1, 2, figsize=(20, 20))  # Create a figure with 1 row and 2 columns
            
            axes[0].imshow(segment_particles(patch), cmap='gray')
            axes[0].set_title(f"{i}th patch centered at {(x_pos,y_pos)}")
            axes[0].axis('off')  # Hide the axes for better visualization
            
            axes[1].imshow(segment_particles(images[idx]), cmap='gray')
            axes[1].set_title(f"{idx}th image")
            axes[1].axis('off')
            
            fig.suptitle(f"Comparison of {i}th patch and {idx}th image", fontsize=14)
            # May be we should add positional encoding as well
            # Show the figure
            plt.show()

feature_matrix = get_feature_matrix(images)
learned_coeff = regress_features(feature_matrix)
U,_,Vh = get_singular_value_decomp(learned_coeff)
left_spanning, right_spanning = spanning_left_right_SV(U,Vh)
visualize_dimensionally_reduced_feature(feature_matrix,left_spanning,right_spanning)


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
    # contours = measure.find_contours(labeled_image, level=0.8)
    # image_separator(labeled_image,i)
    # # visualize_results(segmented_image, regions)
    edge_file_name = "detected_edges" + str(i) + ".png"
    # plot_contours(labeled_image, contours)
    # edge_detector(labeled_image, output_filename = edge_file_name)
    
# For debugging
def visualize_intermediate_steps(original_image, preprocessed_image, enhanced_image, segmented_image, edges_image, final_image, regions):
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    # Plot the different images
    ax[0, 0].imshow(original_image, cmap='gray')
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    ax[0, 1].imshow(preprocessed_image, cmap='gray')
    ax[0, 1].set_title('Preprocessed Image')
    ax[0, 1].axis('off')

    ax[1, 0].imshow(enhanced_image, cmap='gray')
    ax[1, 0].set_title('Enhanced Image')
    ax[1, 0].axis('off')

    ax[1, 1].imshow(segmented_image, cmap='gray')
    ax[1, 1].set_title('Segmented Image')
    ax[1, 1].axis('off')

    ax[2, 0].imshow(edges_image, cmap='gray')
    ax[2, 0].set_title('Edges Image')
    ax[2, 0].axis('off')

    ax[2, 1].imshow(original_image, cmap='gray')
    ax[2, 1].set_title('Final Results')
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax[2, 1].add_patch(rect)
        print("The area of t")
    ax[2, 1].axis('off')

    plt.tight_layout()
    plt.show()
