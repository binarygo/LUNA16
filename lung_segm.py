import numpy as np

from scipy import ndimage
from skimage import measure
from skimage import morphology
from skimage import segmentation
from skimage import filters


def _dilate_mask_impl(mask, ball_r):
    return morphology.binary_dilation(mask, morphology.ball(ball_r))


def _dilate_mask(mask, spacing):
    return _dilate_mask_impl(mask, 10.0 / spacing)


def _find_lung_3d(mask):
    binary = mask

    labels = measure.label(binary)    
    vals, counts = np.unique(labels, return_counts=True)
    counts = counts[vals != 0]
    vals = vals[vals != 0]
    counts_and_vals = sorted(zip(counts, vals), key=lambda x: -x[0])
    
    if len(counts) == 0:
        return None
    
    max_label_counts, max_label = counts_and_vals[0]
    if len(counts) == 1:
        binary[labels!=max_label] = 0
        return binary
    
    max2_label_counts, max2_label = counts_and_vals[1]
    if max2_label_counts > 0.2 * max_label_counts:
        binary[(labels!=max_label)&(labels!=max2_label)] = 0
    else:
        binary[labels!=max_label] = 0
    return binary


def segm_2d(image, spacing=1.0):
    """This funtion segments the lungs from the given 2D slice."""
    # Step 1: Convert into a binary image.
    binary = image < -400
    # Step 2: Remove the blobs connected to the border of the image.
    cleared = segmentation.clear_border(binary)
    # Step 3: Label the image.
    label_image = measure.label(cleared)
    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Step 5: Erosion operation with a disk of radius 2. This operation is 
    #         seperating the lung nodules attached to the blood vessels.
    selem = morphology.disk(2.0 / spacing)
    binary = morphology.binary_erosion(binary, selem)
    # Step 6: Closure operation with a disk of radius 10. This operation is 
    #         to keep nodules attached to the lung wall.
    selem = morphology.disk(10.0 / spacing)
    binary = morphology.binary_closing(binary, selem)
    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = filters.roberts(binary)
    binary = ndimage.binary_fill_holes(edges)

    return binary


def segm_3d(image, spacing=1.0):
    binary = np.stack([
        segm_2d(im, spacing)
        for im in image
    ])
    return _dilate_mask(_find_lung_3d(binary), spacing)
