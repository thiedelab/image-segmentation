import sklearn
import skimage
from skimage import filters, exposure, measure, morphology,segmentation,color,graph, io
from skimage.segmentation import watershed, random_walker
from skimage.feature import canny, peak_local_max
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction import image as m_image
from scipy import ndimage as ndi
import numpy as np
from sklearn.cluster import spectral_clustering
import pims
# It is not drawing on the segmented image. Must fix!
data_path = "/home/yjn2/rProj/data/f3 -0.1V.tif"

# Open the image sequence
raw_frames = pims.open(data_path)

# Creating a list of images
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
# Suppose we try it on a subtracted
def preprocess_image(image):
    # Apply Gaussian filter for denoising
    denoised_image = gaussian_filter(image, sigma=2)
    return denoised_image

def segment_particles(image):
    # Apply manual thresholding
    binary_image = image >= 0.36
    
    # Morphological closing to fill gaps within particles
    closed_image = morphology.binary_closing(binary_image)
    
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


def edge_detector(image, output_filename='detected_edges.png'):
    # Find contours in the binary image
    contours = measure.find_contours(image, level=0.5)

    # Create an image copy to draw on the contours
    output_image = np.zeros_like(image)
    # We may filter heights, to find the relevant height only
    heights = []

    for contour in contours:
        # Draw the contour on the output image
        for point in contour:
            output_image[int(point[0]), int(point[1])] = 1  # Mark the contour points
        
        # Calculate the height of the contour
        height = np.max(contour[:, 0]) - np.min(contour[:, 0])  # Height in the y-axis
        heights.append(height)
    return heights
      

def image_separator(image, index,output_filename = "sepatated.png",):
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background

    # Copied from scipy: Need to translate it some how to make sure I understand it
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

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
    plt.savefig("separated.png")
    plt.close()  # Close the figure to free up memory   


def calculate_area_in_image(image):
    sum = np.sum(image)
    return sum

def spectral_cluster_of_img(img,i):
    mask = img.astype(bool)
    graph = m_image.img_to_graph(img, mask = mask)
    print("Helllooo")
    # Must write a comment explainign this mathematical exp, and why we apply it
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


# Example usage:
for i, image in enumerate(images):
    # # # Step 1: Apply manual thresholding - easier to experiment with
    # enhanced_image = image >= 0.41
    
    # # Step 2: Use edge detection for segmentation
    # segmented_image = segment_particles(enhanced_image)
    
    # # Step 3: Label and filter regions
    # labeled_image, regions = label_and_filter_regions(segmented_image)
    # # Step 4: Visualize the results
    # # contours = measure.find_contours(labeled_image, level=0.8)
    # image_separator(labeled_image,i)
    # # visualize_results(segmented_image, regions)
    if i <= 10:
        spectral_cluster_of_img(image,i)

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







    
# I dont really remember how it works. May be review it if it means anything
# def draw_polygons_on_segmented_image(segmented_image, contours):
#     """Draw approximated polygons on the segmented image."""
#     # Convert segmented_image to an 8-bit format for visualization
#     image_for_drawing = img_as_ubyte(segmented_image)

#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(image_for_drawing, cmap='gray')

#     for contour in contours:
#         # Approximate the contours to polygons
#         epsilon = 0.05 * np.mean([np.linalg.norm(contour[i] - contour[i - 1]) for i in range(1, len(contour))])
#         approx = measure.approximate_polygon(contour, epsilon)
        
#         # Draw the approximated polygon on the image
#         rr, cc = polygon(approx[:, 0], approx[:, 1], segmented_image.shape)
#         mask = np.zeros(segmented_image.shape, dtype=np.uint8)
#         mask[rr, cc] = 1
        
#         # Combine the mask with the original image
#         image_with_polygons = np.copy(image_for_drawing)
#         image_with_polygons[mask > 0] = 255  # Make the polygon visible

#         # Overlay the polygons
#         ax.imshow(image_with_polygons, cmap='gray', alpha=0.5)
#         ax.plot(approx[:, 1], approx[:, 0], linewidth=2, color='red')
    
#     plt.show()

# # Visualization
    # plt.figure(figsize=(12, 6))

    # # Show the original binarized image
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Image')
    # plt.axis('off')

    # # Show the traced contours
    # plt.subplot(1, 2, 2)
    # plt.imshow(output_image, cmap='gray')
    # plt.title('Detected Edges')
    # plt.axis('off')

    # # Save the figure to a file
    # plt.savefig(output_filename)
    # plt.close()  # Close the figure to free up memory  