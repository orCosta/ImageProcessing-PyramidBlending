import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import os

DIV_X_KERNEL = np.array([1, 0, -1])
DIV_Y_KERNEL = DIV_X_KERNEL.reshape(3, 1)
RGB_DIM = 3
GRAY_SCALE = 1
MAX_GRAY_I = 255
FILTER_BASE = np.array([1, 1])


def relpath(filename):
    '''
    :return: the full path of the file
    '''

    return os.path.join(os.path.dirname(__file__), filename)

def read_image(filename, representation):
    '''
    Reads an image file and converts it into a given representation.
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be a
    grayscale image (1) or an RGB image (2).
    :return: normalized image in float64 type.
    '''
    im = imread(filename)
    if im.ndim == RGB_DIM and representation == GRAY_SCALE:
        im = rgb2gray(im)
        return im
    im = im.astype(np.float64)
    im /= MAX_GRAY_I
    return im


def gaussian_filter(filter_size):
    '''
    return 1D gaussian filter, contain approximation of the gaussian distribution.
    :param filter_size: is the size of the gaussian filter.
    :return: 1D gaussian filter.
    '''
    if filter_size == 1:
        return np.array([1])
    fi = np.copy(FILTER_BASE)
    for i in range(filter_size - 2):
        fi = np.convolve(fi, FILTER_BASE)

    filter_vec = np.reshape(fi * (1 / np.sum(fi)), (1, fi.shape[0]))
    return filter_vec


def blur_spatial(im, filter_vec):
    '''
    Performs image blurring using a gaussian filter.
    :param im: the image to blur.
    :param filter_vec: 1D filter for blur.
    :return: The blurry image (float64 image).
    '''
    vertical_filter = np.reshape(filter_vec, (filter_vec.shape[1], 1))
    b_img = convolve(im, np.asmatrix(filter_vec))
    b_img = convolve(b_img, np.asmatrix(vertical_filter))
    return b_img


def reduce_img(img, filter_vec):
    '''
    reduce image by blur it and sub-sampling only every second value and line.
    :param img: the image to reduce.
    :param filter_vec: the filter for blur.
    :return: the reduced image.
    '''
    r_img = np.copy(img)
    r_img = blur_spatial(r_img, filter_vec)
    r_img = np.copy(r_img[::2, ::2])
    return r_img


def expand_img(img, filter_vec):
    '''
    expand image by zero padding and blur.
    :param img:the image to expand.
    :param filter_vec: the filter for blur.
    :return: the expanded image.
    '''
    ex_img = np.zeros((img.shape[0]*2, img.shape[1]*2))
    ex_img[::2, ::2] = img
    ex_img = blur_spatial(ex_img, 2 * filter_vec)
    return ex_img


def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    construct a Gaussian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter to be used in constructing the pyramid filter.
    :return: the pyramid as an array, where each element of the array is a grayscale image.
    and filter vector which is for the pyramid construction.
    '''
    filter_vec = gaussian_filter(filter_size)
    g = np.copy(im)
    pyr = [g]

    for i in range(max_levels -1):
        g = reduce_img(g, filter_vec)
        pyr.append(g)
        if g.shape[0] < 16 or g.shape[1] < 16:
            break

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    construct a Laplacian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter to be used in constructing the pyramid filter.
    :return: the pyramid as an array, where each element of the array is a grayscale image.
    and filter vector which is for the pyramid construction.
    '''
    g_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    l_pyr = []
    for i in range(len(g_pyr)-1):
        g1 = g_pyr[i]
        g2 = g_pyr[i+1]
        l = g1 - expand_img(g2, filter_vec)
        l_pyr.append(l)

    l_pyr.append(g_pyr[-1])

    return l_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    Reconstruct an image from its Laplacian Pyramid.
    :param lpyr:  the Laplacian pyramid.
    :param filter_vec: the filter that was used in constructing the pyramid.
    :param coeff: is a vector. The vector size is the same as the number of levels in the pyramid and
    contain the levels coefficients.
    :return: the image.
    '''
    i = len(lpyr) - 1
    img = lpyr[i] * coeff[i]
    i -= 1
    while i >= 0:
        img = expand_img(img, filter_vec) + (lpyr[i] * coeff[i])
        i -= 1

    return img


def render_pyramid(pyr, levels):
    '''
    creates a single black image in which the pyramid levels of the given
    pyramid pyr are stacked horizontally.
    :param pyr:  Gaussian or Laplacian pyramid.
    :param levels: is the number of levels to present in the result.
    :return: one single image contains all the given levels of the pyramid.
    '''
    tot_height = pyr[0].shape[0]
    tot_width = 0
    for i in range(levels):
        tot_width += pyr[i].shape[1]
        pyr[i] = ((pyr[i] - np.min(pyr[i])) / np.ptp(pyr[i]))

    comb_img = np.zeros((tot_height, tot_width))
    current_row = 0
    current_col = 0
    for i in range(levels):
        height, width = pyr[i].shape[:2]
        comb_img[current_row: current_row + height, current_col:current_col + width] = pyr[i]
        current_col += width

    return comb_img


def display_pyramid(pyr, levels):
    '''
    display the stacked pyramid image of the given levels.
    :param pyr: Gaussian or Laplacian pyramid.
    :param levels: is the number of levels to present in the result.
    '''
    comb_img = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(comb_img)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.show()


def to_binary(img, threshold):
    '''
    creates boolean image according to the given threshold.
    :param img: grayscale image.
    :param threshold: the bound to create the boolean values.
    :return: boolean image
    '''
    return (threshold > img)


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    '''
    :param im1: first grayscale image to be blended.
    :param im2: second grayscale image to be blended.
    :param mask: is a boolean mask containing True and False representing which parts
           of im1 and im2 should appear in the resulting im_blend.
    :param max_levels: is the max levels of the Gaussian and Laplacian pyramids that created.
    :param filter_size_im: is the size of the Gaussian filter which defining the filter
           used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask:  is the size of the Gaussian filter which defining the filter
           used in the construction of the Gaussian pyramid of mask.
    :return: the blended image.
    '''
    l1, f_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, f_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    g_mask, f_vec3 = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)

    l_blend = []
    for i in range(len(l1)):
        l_blend.append(g_mask[i] * l1[i] + (1 - g_mask[i]) * l2[i])

    coeff = [1] * max_levels
    im_blend = laplacian_to_image(l_blend, f_vec1, coeff)
    return im_blend


def blending_example1():
    '''
    blending example of RGB images.
    :return:
    '''
    img1 = read_image(relpath('externals/man.jpg'), 2)
    img2 = read_image(relpath('externals/lion.jpg'), 2)
    mask = read_image(relpath('externals/mask1.jpg'), 1)
    mask = to_binary(mask, 0.5)
    mask = mask.astype(np.bool)
    f_size = 7
    mask_filter = 11
    max_levels = 7

    im_blend = np.copy(img1)
    im_blend[:,:,0] = pyramid_blending(img1[:,:,0], img2[:,:,0], mask,max_levels, f_size, mask_filter)
    im_blend[:, :, 1] = pyramid_blending(img1[:, :, 1], img2[:, :, 1], mask, max_levels, f_size, mask_filter)
    im_blend[:, :, 2] = pyramid_blending(img1[:, :, 2], img2[:, :, 2], mask, max_levels, f_size, mask_filter)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(mask)
    plt.axis('off')
    plt.set_cmap('gray')
    plt.subplot(2, 2, 4)
    plt.imshow(np.clip(im_blend,0,1))
    plt.axis('off')
    plt.show()
    return img1, img2, mask, im_blend


def blending_example2():
    '''
    blending example of RGB images.
    :return:
    '''
    img1 = read_image(relpath('externals/huji.jpg'), 2)
    img2 = read_image(relpath('externals/game_of.jpg'), 2)
    mask = read_image(relpath('externals/mask2.jpg'), 1)
    mask = to_binary(mask, 0.5)
    mask = mask.astype(np.bool)
    f_size = 7
    max_levels = 3

    im_blend = np.copy(img1)
    im_blend[:,:,0] = pyramid_blending(img1[:,:,0], img2[:,:,0], mask,max_levels, f_size, f_size)
    im_blend[:, :, 1] = pyramid_blending(img1[:, :, 1], img2[:, :, 1], mask, max_levels, f_size, f_size)
    im_blend[:, :, 2] = pyramid_blending(img1[:, :, 2], img2[:, :, 2], mask, max_levels, f_size, f_size)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(mask)
    plt.axis('off')
    plt.set_cmap('gray')
    plt.subplot(2, 2, 4)
    plt.imshow(np.clip(im_blend, 0, 1))
    plt.axis('off')
    plt.show()
    return img1, img2, mask, im_blend


