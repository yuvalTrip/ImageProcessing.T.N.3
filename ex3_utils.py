import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 318916335


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points' location [[x,y]...], [[dU,dV]...] for each points
    """
    # I used:
    # https://www.youtube.com/watch?v=yFX_N5p0kO0
    # https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
    # https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/
    # https://www.youtube.com/watch?v=6wMoHgpVUn8&t=326s
    original_x_y = []  # Original points' location list
    u_v = []  # Change for each point in x&y axis
    half_win_size = win_size // 2  # window_size is odd(as given), all the pixels with offset in between [-w, w] are inside the window
    kernel = np.array([[1, 0, -1]])  # the kernel we used to compute the derivative
    I_x = cv2.filter2D(im2, -1, kernel, cv2.BORDER_REPLICATE)  # The derivative of I2 in the x-axis
    I_y = cv2.filter2D(im2, -1, kernel.T, cv2.BORDER_REPLICATE)  # The derivative of I2 in the y-axis
    I_t = im2 - im1  # The derivative of I by t
    hight, width = I_x.shape[:2]
    # Move over all pixels in image, building the window,and check the pixels according to the step size.
    for i in range(half_win_size, hight - half_win_size + 1, step_size):
        for j in range(half_win_size, width - half_win_size + 1, step_size):
            mat_y = I_y[i - half_win_size:i + half_win_size + 1, j - half_win_size:j + half_win_size + 1]  # compute y matrix
            mat_x = I_x[i - half_win_size:i + half_win_size + 1, j - half_win_size:j + half_win_size + 1]  # compute x matrix
            # Compute ATA matrix . Means:
            # ATA= [sigma(Ix)^2 ,sigma(IxIy), sigma(IyIx) sigma(Iy)^2]
            ATAmat = np.array([[np.sum(mat_x ** 2), np.sum(mat_x * mat_y)],  # compute xy matrix
                               [np.sum(mat_x * mat_y), np.sum(mat_y ** 2)]])
            # Compute the eigenvalues of ATA matrix.
            lambdas = np.linalg.eigvals(ATAmat)
            # according the condition as we learned in tirgul and written in PDF
            if lambdas.min() > 1 and lambdas.max() / lambdas.min() < 100:
                mat_t = I_t[i - half_win_size:i + half_win_size + 1, j - half_win_size:j + half_win_size + 1]  # compute t matrix
                # Compute the matrix ATb=[-sigma(IxIt), -sigma(IyIt)]
                ATbmat = np.array([-np.sum(mat_x * mat_t), -np.sum(mat_y * mat_t)])
                # Compute the (multiplicative) inverse of ATA matrix and than multiply it with ATb, i.e.:
                # v=(AtA)^-1 *ATb, v=[Vx,Vy]=[du,dv]=[delta x/delta t,delta y/delta t]
                local_im_flow_vector_uv = np.linalg.inv(ATAmat).dot(ATbmat)
                # multiply by -1
                local_im_flow_vector_uv = local_im_flow_vector_uv * (-1)
                u_v.append(local_im_flow_vector_uv)  # add local_im_flow_vector_uv to uv array
                original_x_y.append([j, i])  # add points location to original_points array
    # Convert to np array
    original_x_y = np.array(original_x_y)
    u_v = np.array(u_v)
    return original_x_y, u_v


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid for a given image
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    # I used the links:
    # https://paperswithcode.com/method/laplacian-pyramid
    # https://www.youtube.com/watch?v=D3-IK-y9UN0
    pyramid_laplacian_list = []  # Initialize a list of all images in pyramid
    gauss_pyr = gaussianPyr(img, levels)  # Create gaussian pyramid

    gaussian = gaussianKer(5)#Cefine the kernel
    for i in range(1, levels):
        # Expand the image in i+1 index from pyramid_list_gaussian i.e. expands a gaussian pyramid level one step up
        expand = gaussExpand(gauss_pyr[i], gaussian)
        lap_im = gauss_pyr[i - 1] - expand  # Create the difference image i.e. subtraction between gaussianPyr-x-Expand gaussianPyr x + 1
        pyramid_laplacian_list.append(lap_im)#Add the new laplacian filtered image to the pyramid

    pyramid_laplacian_list.append(gauss_pyr[levels - 1])#Add the last laplacian filtered image to the pyramid

    return pyramid_laplacian_list


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gaussian = gaussianKer(5) # I did not use the same way as I defined the gaussian kernel before because it did not worked well.
    n = len(lap_pyr) - 1
    gauss_Pyr = lap_pyr[n]

    for i in range(n, 0, -1):
        expand = gaussExpand(gauss_Pyr, gaussian)
        gauss_Pyr = expand + lap_pyr[i - 1]
    return gauss_Pyr


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    # img = cropPic(img, levels)
    # pyrLst = [img]
    # gaussian = gaussianKer(5)
    #
    # for i in range(1, levels):
    #     I_temp = cv2.filter2D(pyrLst[i - 1], -1, gaussian, cv2.BORDER_REPLICATE)
    #     I_temp = I_temp[::2, ::2]
    #     pyrLst.append(I_temp)
    # return pyrLst
    # I use the links:
    # https://www.youtube.com/watch?v=dW7sMgs-Ggw
    # the presentation from tirgul
    pyramid_list = []  # initialize a list of all images in pyramid
    # in case of RGB or grayscale image:
    #############################################################
    hight, width = img.shape[:2]
    hight = (2 ** levels) * (hight // (2 ** levels))
    width = (2 ** levels) * (width // (2 ** levels))
    # hight = (2**levels) * np.floor(hight / 2**levels).astype(np.uint8)#(hight // (2**levels))
    #     width = (2**levels) * np.floor(width / 2**levels).astype(np.uint8)
    if img.ndim == 3:  # if image is RGB
        img = img[0:hight, 0:width, :]
    else:  # if image is grayscale
        img = img[0:hight, 0:width]

    pyramid_list.append(img)  # add original image to the pyramid
    k_size = 5  # as define before in T.N.2
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8  # as it has been written in PDF

    # by using pythons internal function 'getGaussianKernel', we will creates Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(k_size, sigma)
    # gaussian_kernel=gaussianKer(5)
    for i in range(1, levels):
        # blur the image with a Gaussian kernel
        # (by using pythons internal function 'filter2D', we will apply the gaussian_kernel on the last image in pyramid_list )
        Itemp = cv2.filter2D(pyramid_list[i - 1], -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)
        # Sample every second pixel
        Itemp = Itemp[::2, ::2]

        pyramid_list.append(Itemp)  # add the new gaussian filtered image to the pyramid
    return pyramid_list


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
     Function get image in k level from gaussian pyramid and return image from gaussian pyramid in level k-1
    :param img: Image in k level from gaussian pyramid
           gs_k: The kernel we use for expanding
    :return: Image from gaussian pyramid in level k-1 (i.e. the expend level)
    """
    gs_k = (gs_k / gs_k.sum()) * 4 # As we learned in class
    if img.ndim == 3:
        hight, width, depth = img.shape[:3]
        zero_mat = np.zeros((2 * hight, 2 * width, depth))
    else:
        hight, width = img.shape[:2]
        zero_mat = np.zeros((2 * hight, 2 * width))
    # Samples every second pixel
    zero_mat[::2, ::2] = img
    # Blur the image with a Gaussian kernel
    # (by using pythons internal function 'filter2D', we will apply the gaussian_kernel on the zero padded image )
    image = cv2.filter2D(zero_mat, -1, gs_k, cv2.BORDER_REPLICATE)
    return image


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    # Links I used:
    # https://theailearner.com/tag/image-blending-with-pyramid-and-mask/

    # Build the laplacian pyramid for both the images
    lap_pyr1 = laplaceianReduce(img_1, levels)
    lap_pyr2 = laplaceianReduce(img_2, levels)
    # Build Gaussian pyramid for the mask
    gauss_pyr_mask = gaussianPyr(mask)

    merge = (lap_pyr1[levels - 1] * gauss_pyr_mask[levels - 1]) + ((1 - gauss_pyr_mask[levels - 1]) * lap_pyr2[levels - 1])
    gaussian = gaussianKer(5)  # I did not use the same way as I defined the gaussian kernel before because it did not worked well.
    for i in range(levels - 2, -1, -1):
        merge = gaussExpand(merge, gaussian)
        merge = merge + (lap_pyr1[i] * gauss_pyr_mask[i]) + ((1 - gauss_pyr_mask[i]) * lap_pyr2[i])

    img_1 = cropSizeImage(img_1, levels)
    img_2 = cropSizeImage(img_2, levels)
    naive = (img_1 * gauss_pyr_mask[0]) + ((1 - gauss_pyr_mask[0]) * img_2)

    return naive, merge


def cropSizeImage(img: np.ndarray, levels: int) -> np.ndarray:
    twoPowLevel = pow(2, levels)
    h, w = img.shape[:2]
    h = twoPowLevel * np.floor(h / twoPowLevel).astype(np.uint8)
    w = twoPowLevel * np.floor(w / twoPowLevel).astype(np.uint8)

    if img.ndim == 3:
        img = img[0:h, 0:w, :]
    else:
        img = img[0:h, 0:w]
    return img


# def gaussianKer(kernel_size: int) -> np.ndarray:
#     gaussian = cv2.getGaussianKernel(kernel_size, -1)
#     gaussian = gaussian.dot(gaussian.T)
#     return gaussian

def gaussianKer(kernel_size: int) -> np.ndarray:
    """
    (Function I took from Stackoverflow)
    Compute the gaussian kernel for given k_size
    :param kernel_size:
    :return:
    """
    gaussian = cv2.getGaussianKernel(kernel_size, -1)
    gaussian = gaussian / gaussian[0, 0]
    gaussian_kernel = gaussian @ gaussian.T
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel
