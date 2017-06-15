'''
Utilities
'''
import math
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog as skhog

from lane_config import config as lane_config
from vehicle_config import config as vehicle_config

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def normalize(img):
    '''
    Normalize the grayscale image

    Arguments:
    a: the grayscale image to normalize
    Return: the normalized grayscale image
    '''
    if img.dtype != np.float32:
        img = np.array(img, dtype=np.float32)
    low = np.amin(img, axis=(0, 1))
    high = np.amax(img, axis=(0, 1))
    mid = (high + low) * 0.5
    dis = (high - low + 0.1) * 0.5  # +0.1 in case min = max
    return (img - mid) / dis

def grayscale(img):
    '''
    Convert the image to grayscale

    Arguments:
    img: the image to convert to grayscale
    Return: the converted grayscale image

    '''
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

def convert_color(image, cspace):
    '''
    Convert the image's colorspace and normalize pixel to to range 0.0, 1.0
    image: the image
    cspace: the color space
    '''
    if "max" not in convert_color.__dict__:
        convert_color.max = np.amax(image)
        if convert_color.max < 2: # the max pxe value should be 0 t 1
            convert_color.max = 1
        else: # the max pxe value should be 0 t0 255
            convert_color.max = 255.0

    # Normalize image first so the valus fall between 0 and 1
    if convert_color.max == 1:
        image = np.uint8(image * 255.0)
    if cspace != 'RGB':
        if cspace == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif cspace == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def to_rgb(image, cspace):
    '''
    Convert the image's colorspace and normalize pixel to to range 0.0, 1.0
    image: the image
    cspace: the color space
    '''
    if cspace != 'RGB':
        if cspace == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif cspace == 'LUV':
            image = cv2.cvtColor(image, cv2.COLOR_LUV2RGB)
        elif cspace == 'HLS':
            image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
        elif cspace == 'YUV':
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
        elif cspace == 'YCrCb':
            image = cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
        elif cspace == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image/255.0
    return image

def find_houghlines(image, dest=None, rho=1, theta=np.pi/180, threshold=25, min_line_len=100, max_line_gap=40,
                    thickness=2, slop_threshold=1.963, angle=None):
    """
    image should be the output of a sobel transform.
    Returns an image with hough lines drawn.
    """
    if angle is not None:
        slop_threshold = math.cos(0.15 * np.pi) / math.sin(0.15 * np.pi)

    def fitline(lines, previous=None):
        '''
        Fit the lines with a first order polynominal
        Return: (ymin, f) where ymin is the minimal y coordinate of the lines, f is the polynominal function

        Parameters:
        lines: the lines
        previous: result of the previous fit
        '''
        if len(lines) == 0:
            return previous
        x = [a[0] for a in lines] + [a[2] for a in lines]
        y = [a[1] for a in lines] + [a[3] for a in lines]

        # weight points by their line length, penaltize relatively short lines
        ylen = [abs(a[1] - a[3]) for a in lines]
        minlen = np.min(ylen)
        maxlen = np.max(ylen)
        if maxlen - minlen < 1:
            w = None
        else:
            w = np.exp((ylen - minlen)/(maxlen - minlen) + 1.0)
            w = np.repeat(w, 2) # each weight value is for two end points

        if len(x) > 3:
            z = np.polyfit(y[:], x, 2, w=w) # weighted polynominal fit of the points
        else:
            z = np.polyfit(y[:], x, 1, w=w) # weighted polynominal fit of the points
        f = np.poly1d(z)
        return np.min(y), f

    def interpolate(lines, top, bottom, step):
        '''
        Interpolate the lines by fitting the end points with a linear polynomial function.
        bottom: specify bottom of the image, the line will extend to the bottom of the image
        the same as bottom so there is no extension.
        '''
        fits = fitline(lines)
        points = []
        if fits is None:
            return None
        for pos in range(top, bottom, step):
            points.append([int(fits[1](pos)), pos])
        return points

    def draw_lines(image, lines):
        '''
        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        '''
        if lines is None:
            return
        # filter the lines to exclude lines that are nearly horizontal
        filtered_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if math.fabs(y2 - y1) > 0:
                    m = (x2 - x1) / (y2 - y1)
                    if math.fabs(m) < slop_threshold:
                        filtered_lines += [[x1, y1, x2, y2]]
                        cv2.line(image, (x1, y1), (x2, y2), 255, thickness)

    # detect_houghlines starts here
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    if lines is not None:
        if dest is None:
            dest = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        draw_lines(dest, lines)
        return dest
    else:
        return dest

def draw_lane(shape, top, bottom, step, left, right):
    # Create an image to draw the lines on
    image = np.zeros(shape, dtype=np.uint8)
    left_points = []
    right_points = []
    for i in range(top, bottom, step):
        if lane_config.bestfit and left.best_fit is not None:
            left_points.append([int(left.bestx(i)), i])
        else:
            left_points.append([int(left.x(i)), i])
        if lane_config.bestfit and right.best_fit is not None:
            right_points.append([int(right.bestx(i)), i])
        else:
            right_points.append([int(right.x(i)), i])
    pts = np.vstack((np.array(left_points), np.array(right_points)[::-1]))
    if lane_config.test:
        print(left_points)
        print(right_points)
        print(pts)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(image, np.int_([pts]), (0, 255, 0))
    return image

def line_mask(image_or_shape, line, top, bottom, step, top_width, bottom_width):
    '''
    Create a line mask
    '''
    if isinstance(image_or_shape, tuple):
        image = np.zeros((bottom - top, image_or_shape[1]), dtype=np.uint8)
    else:
        image = image_or_shape
    left_points = []
    right_points = []
    grad = (bottom_width - top_width) / (bottom - top)
    for i in range(top, bottom, step):
        if lane_config.bestfit:
            x = int(line.bestx(i))
        else:
            x = int(line.x(i))
        w = grad * (i - top) + top_width
        left_points.append([int(x-w), i - top])
        right_points.append([int(x+w), i - top])

    pts = np.vstack((np.array(left_points), np.array(right_points)[::-1]))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(image, np.int_([pts]), 255)
    return image

def weighted_img(image, overlay, α=0.8, β=0.2, λ=0.):
    """
    `overlay` is the overlay
    `image` should be the image before any processing.
    The result image is computed as follows:
    image * α + overlay * β + λ
    NOTE: image and ioverlaymg must be the same shape!
    """
    return cv2.addWeighted(image, α, overlay, β, λ)

def bin_spatial(image, size=(32, 32)):
    '''
    Compute binned color features
    image: the image
    size: image size to scale
    '''
    return cv2.resize(image, size).ravel()

def color_histogram(image, bins=32, range=(0, 256)):
    '''
    Compute the histogram of the color channels separately
    image: the image
    bins: the number of bins
    range: bin's value range
    '''
    channel1 = np.histogram(image[:, :, 0], bins=bins, range=range)
    channel2 = np.histogram(image[:, :, 1], bins=bins, range=range)
    channel3 = np.histogram(image[:, :, 2], bins=bins, range=range)
    # Concatenate the histograms into a single feature vector
    return np.concatenate((channel1[0], channel2[0], channel3[0]))

def hog(image, orientations, pixels_per_cell, cells_per_block, visualise=False, feature_vector=True):
    '''
    Extract hog from the image
    image: the image
    orientations: hog orientation to extract
    pixels_per_cell: number of pixel per hog cell
    cells_per_block: number of cells per hog block
    visualise: True to return hog_image, False to return None
    feature_vector: True to return the future vector, False to return the flatten features
    '''
    if visualise:
        return skhog(image, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                     cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=True, block_norm='L2-Hys',
                     visualise=visualise, feature_vector=feature_vector)
    else:
        return skhog(image, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                     cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=True, block_norm='L2-Hys',
                     visualise=visualise, feature_vector=feature_vector), None

def show(images, dim=None, titles=None, file=None):
    '''
    Show images
    images the images to show
    dim: diomension of the view in terms of number of rwos and columns
    titles: titles of the images
    file: Save the plot in the file instead of showing on screen
    '''
    if dim is None:
        if len(images) <= 4:
            dim = [1, len(images)]
        elif len(images) <= 16:
            dim = [int(math.ceil(len(images) / 4.)), 8]
        else:
            dim = [int(math.ceil(len(images) / 8.)), 8]

    fig, axes = plt.subplots(dim[0], dim[1], figsize=(16, 8))
    fig.tight_layout(pad=10, w_pad=10, h_pad=5.0)

    if dim[0] == 1:
        for column in range(dim[1]):
            if column >= len(images):
                break
            axes[column].imshow(images[column])
            if titles is not None:
                axes[column].set_title(titles[column], fontsize=8)
    else:
        for row in range(dim[0]):
            for column in range(dim[1]):
                index = row * dim[1] + column
                if index >= len(images):
                    break
                axes[row][column].imshow(images[index])
                if titles is not None:
                    axes[row][column].set_title(titles[index], fontsize=8)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95, wspace=0.1, hspace=0.1)
    if file is None:
        plt.show()
    else:
        plt.savefig(file, bbox_inches='tight')
