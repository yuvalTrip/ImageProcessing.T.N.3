import math
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
# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


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
            # according the conditions as we learned in tirgul and written in PDF
            if lambdas.min() > 1 and lambdas.max() / lambdas.min() < 100 and lambdas.max()>=lambdas.min():
                mat_t = I_t[i - half_win_size:i + half_win_size + 1, j - half_win_size:j + half_win_size + 1]  # compute t matrix
                # Compute the matrix ATb=[-sigma(IxIt), -sigma(IyIt)]
                ATbmat = np.array([-np.sum(mat_x * mat_t), -np.sum(mat_y * mat_t)])
                # Compute the (multiplicative) inverse of ATA matrix and than multiply it with ATb, i.e.:
                # v=(AtA)^-1 *ATb, v=[Vx,Vy]=[du,dv]=[delta x/delta t,delta y/delta t]
                local_im_flow_vector_uv = np.linalg.inv(ATAmat).dot(ATbmat)
                # multiply by -1
                local_im_flow_vector_uv = local_im_flow_vector_uv * (-1)
                u_v.append(local_im_flow_vector_uv)  # add local_im_flow_vector_uv (i.e. movement vector ) to uv array
                original_x_y.append([j, i])  # add points location to original_points array
    # Convert to np array
    original_x_y = np.array(original_x_y)
    u_v = np.array(u_v)
    return original_x_y, u_v

def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pass


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    # Define array of error we will use later
    mean_errors=[]
    # Define array of new temp values of X and Y cordinates
    temp_cordinates=[]
    # We will take the size of the image
    height, width = im1.shape[:2]
    # Get U_V array from lucas kanade
    x_y, u_v= opticalFlow(im1 ,im2,10)
    # We will compute the cordinate of im2, i.e., the cordinate of im1 after translation
    # np.round because u_v contains float values
    im2_cordinate=np.round(x_y+u_v)
    # For each value in u_v
    for value in u_v:
        # For each pixel in image
        for i in range (1,height):
            for j in range(1, width - 1):
                # If the pixel moved in LK
                if (i,j) in x_y:
                    xy_tmp_new=[np.round(j+value[0]),np.round(i+value[1])]
                    temp_cordinates.append(xy_tmp_new)
            # # We will translate the image according to U_V values
            # mat_after_translation=
        # Define new matrix represent the error between first image after translation according to specific U_V to img2
        error_mat=abs(im2_cordinate-temp_cordinates)
        # Compute the average error
        average_error=error_mat.mean()
        # Add the mean error to mean_errors array
        mean_errors.append(average_error)
    # Convert to np array
    mean_errors = np.array(mean_errors)
    # From all mean errors we computed, we will select the lowes error
    min_error=min(mean_errors)
    # Find the index of min_error
    result = np.where(mean_errors == min_error)
    # Save the appropriate value of U_V (assume that min_error is unique value)
    most_accurate_uv= u_v[result]
    # define the translation matrix (as written here: https://en.wikipedia.org/wiki/Translation_(geometry) )
    translation_matrix=np.zeros((3,3))
    translation_matrix[0,0]=1
    translation_matrix[1,1]=1
    translation_matrix[2,2]=1 # The rest will be zeros
    translation_matrix[0,2]=most_accurate_uv[0]
    translation_matrix[1,2]=most_accurate_uv[1]
    # Return the translation matrix:
    return translation_matrix





def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    # Remember: Rigid is combination of translation and rotation

    # We will take the size of the image
    height, width = im1.shape[:2]
    # Get U_V array from lucas kanade
    x_y, u_v = opticalFlow(im1, im2, 10)
    # Now we will use the formulla of Rigid Transformations as we learned:
    # [x',y']=[[cos(teta), -sin(teta), u],[sin(teta),cos(teta),v]] *[x,y,1]
    # For each value in u_v
    for value in u_v:
        # Compute the matrix:
        # [[cos(teta), -sin(teta), u],[sin(teta),cos(teta),v]]
        first_mat=np.array([[math.cos(teta), -math.sin(teta),value[0]],[math.sin(teta),math.cos(teta),value[1]]])
        # For each pixel in image
        for i in range(1, height):
            for j in range(1, width - 1):
        # If the pixel moved in LK
                if(i,j) in x_y:
                    # Compute the matrix:
                    # [x,y,1]
                    sec_mat=np.array([i,j,1])



    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------

def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
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

# ---------------------------------------------------------------------------
# --------------------- Auxiliary functions  --------------------------------
# ---------------------------------------------------------------------------

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
