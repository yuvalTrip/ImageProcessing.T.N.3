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
            if lambdas.min() > 1 and lambdas.max() / lambdas.min() < 100 and lambdas.max() >= lambdas.min():
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


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    # I used the following links:
    # https://docs.opencv.org/3.4/d5/d0f/tutorial_js_pyramids.html
    # https://www.youtube.com/watch?v=i03K_tOwtZ8
    # https://www.youtube.com/watch?v=OdElW_aMrzI
#     if img1.ndim > 2: # In case image is RGB
#         img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#         img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Create gaussian pyramids for both images
    im1_pyr = gaussianPyr(img1, k)
    im2_pyr = gaussianPyr(img2, k)
    # Create 3d array, with a shape of (m, n, 2), where the first channel will hold U, and the second V- as we required to return
    temp = np.zeros((im1_pyr[k - 2].shape[0], im1_pyr[k - 2].shape[1], 2))
    last = np.zeros((im1_pyr[k - 1].shape[0], im1_pyr[k - 1].shape[1], 2))
    # Compute the opticalFlow for the smallest image
    x_y, u_v = opticalFlow(im1_pyr[k - 1], im2_pyr[k - 1], stepSize, winSize)
    for j in range(len(x_y)):  # change pixels uv by formula
        u, v = u_v[j]# Extract u,v values
        y, x = x_y[j] # Extract x,y values
        last[x, y, 0] = u# Fill the first channel with u values
        last[x, y, 1] = v# Fill the second channel with v values
    # for each level of both pyramids from small img to big
    for i in range(k - 2, -1, -1):
        # Compute the x_y and u_v for both images
        x_y, u_v = opticalFlow(im1_pyr[i], im2_pyr[i], stepSize, winSize)
        for j in range(len(x_y)):  # change pixels uv by formula
            u, v = u_v[j]  # Extract u,v values
            y, x = x_y[j]# Extract x,y value
            temp[x, y, 0] = u # Fill the first channel with u values
            temp[x, y, 1] = v # Fill the second channel with v values
        for index1 in range(last.shape[0]):
            for index2 in range(last.shape[1]):
                # compute the new UV by the given formula in PDF: Ui = Ui + 2 ∗ Ui−1, Vi = Vi + 2 ∗ Vi−1
                temp[index1 * 2, index2 * 2, 0] += last[index1, index2, 0] * 2
                temp[index1 * 2, index2 * 2, 1] += last[index1, index2, 1] * 2
        # We will save the temp mat as last mat
        last = temp.copy()
        # And while we are not in last level,
        if i - 1 >= 0:
            # We will initialize again the temp mat as matrix of zeros, before filling it again
            temp.fill(0)
            # Change size before continuing the next level (i.e., different size of image in the pyramid)
            temp.resize((im1_pyr[i - 1].shape[0], im1_pyr[i - 1].shape[1], 2), refcheck=False)

    return temp


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    # I used the link:
    # https://theailearner.com/tag/cv2-warpperspective/

    # Define minimal error to be the biggest int value
    minimal_error = float("inf")
    spot = 0
    # Get U_V array from lucas kanade
    x_y, u_v = opticalFlow(im1, im2, 10, 5)
    # Iterate over all the u_v we found
    for x in range(len(u_v)):
        u = u_v[x][0] #u value
        v = u_v[x][1] #v value
        temp_translation = np.array([[1, 0, u],[0, 1, v],[0, 0, 1]], dtype=np.float)
        # Now we will applies a perspective transformation to an im1 to temp_translation by using (im1.shape[1], im1.shape[0])-i.e. create a new image a transformation using u,v
        newimg = cv2.warpPerspective(im1, temp_translation, (im1.shape[1], im1.shape[0]))
        # We will compute the difference between im2 to im1 after applying the transformation
        temp_error = ((im2 - newimg) ** 2).sum()

        # We will find the most accurate u_v according to the smallest error
        if temp_error < minimal_error:# If temp_error is smaller than min_error
            minimal_error = temp_error # We will define the new error to be the new min_error
            spot = x
            # If there is no change between both images
            if minimal_error == 0:
                print("break-images are the same!")
                break
    u = u_v[spot][0]
    v = u_v[spot][1]

    temp_translation = np.array([[1, 0, u],
                  [0, 1, v],
                  [0, 0, 1]], dtype=np.float)

    return temp_translation

# #####MINE
def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """

    # Remember: Rigid is combination of translation and rotation
    # I used the links:
    # http://16720.courses.cs.cmu.edu/lec/alignment_lec11.pdf
    # presentation number 5, page number 28
    # My algo will look like that:
    # teta0     teta1   ....    teta359
    # | | |     | | |           | | |
    # uv uv uv uv uv uv        uv uv uv
    # From that we will take the uv with the minimum mean error and the appropriate teta to be the teta,tx,ty for the rigid matrix

    # Define minimal error to be the biggest int value
    minimal_error = float("inf")
    spot = 0
    # Get U_V array from lucas kanade
    x_y, u_v = opticalFlow(im1, im2, 10, 5)
    # Iterate over all the u_v we found
    for index in range(len(u_v)):
        u = u_v[index][0]  # u value
        v = u_v[index][1]  # v value
        # Now we will compute the angle between u and v according to next comuting:
        # tan(teta)=v/u  --> teta= arctan(v/u)
        teta=0 # If denominator is 0, than the angle will be 0 too.
        if u!=0:
            teta = np.arctan(v / u)
        # Compute the matrix:
        # [[cos(teta), -sin(teta), u],[sin(teta),cos(teta),v]]
        temp_mat = np.array([[math.cos(teta), -math.sin(teta), u],
                              [math.sin(teta), math.cos(teta), v],
                              [0,0,1]],dtype=np.float)
        # Now we will applies a perspective transformation to an im1 to temp_translation by using (im1.shape[1], im1.shape[0])-i.e. create a new image a transformation using u,v
        newimg=cv2.warpPerspective(im1, temp_mat, (im1.shape[1], im1.shape[0]))
        # We will compute the difference between im2 to im1 after applying the transformation
        temp_error = ((im2 - newimg) ** 2).sum()
        # We will find the most accurate u_v according to the smallest error
        if temp_error < minimal_error:  # If temp_error is smaller than min_error
            minimal_error = temp_error  # We will define the new error to be the new min_error
            spot = index
            # If there is no change between both images
            if minimal_error == 0:
                print("break-images are the same!")
                break
    # with the teta we get the minimal_error we will use findtranslationLK to find the uv
    u = u_v[index][0]  # u value
    v = u_v[index][1]  # v value
    # Now we will compute the angle between u and v according to next comuting:
    # tan(teta)=v/u  --> teta= arctan(v/u)
    teta = 0  # If denominator is 0, than the angle will be 0 too.
    if u != 0:
        teta = np.arctan(v / u)
    # Compute the matrix:
    # [[cos(teta), -sin(teta), u],[sin(teta),cos(teta),v],[0,0,1]]
    temp_mat = np.array([[math.cos(teta), -math.sin(teta), u],
                         [math.sin(teta), math.cos(teta), v],
                         [0, 0, 1]],dtype=np.float)
    # Now we will applies a perspective transformation to an im1 to temp_translation by using (im1.shape[1], im1.shape[0])-i.e. create a new image a transformation using u,v
    newimg = cv2.warpPerspective(im1, temp_mat, (im1.shape[1], im1.shape[0]))
    mat = findTranslationLK(newimg, im2)
    T = mat @ temp_mat
    return (T)


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
    # Links I used:
    # Presentation number 5- backward warping
    # Tirgul lecture 10

    #We will take the size of the image
    height, width = im1.shape[:2]
    # Define new matrix of zeros in the sime shapes:
    newimg=np.zeros((height,width))
    # Calculating the inverse of T matrix
    T_inverse=np.linalg.inv(T)
    if T_inverse[0][1]!=0:
        degree=np.rad2deg(np.arcsin(T_inverse[1][0]))
        radians=np.deg2rad(-degree)
        T_inverse[0][0] = np.cos(radians)
        T_inverse[1][1] = np.cos(radians)
        T_inverse[0][1] = -np.sin(radians)
        T_inverse[1][0] = np.sin(radians)
    # Iterate over all pixels in image
    for j in range(0, height):
        for i in range(0, width):
            # Define array of the new coordinates [x,y,1]
            new_coor = np.array([i, j, 1])
            # We will multiply T_inverse by new_coor and get array in size of 3
            newarr = T_inverse @ new_coor
            # Define new x
            x = newarr[0].astype(int)
            # Define new y
            y = newarr[1].astype(int)
            if 0 <= x < height and 0 <= y < width:
                newimg[i, j] = im1[x, y]
    plt.imshow(newimg)
    plt.show()
    return newimg


    ###################################################
    # # First channel will be the coordinates in x-axis
    # # Second channel will be the coordinates in y-axis
    #
    # # Create 2 marrices in the same size
    # Y_indices = np.zeros((height, width))
    # X_indices = np.zeros((height, width))
    # # For each pixel in image
    # for j in range(0, height):
    #     k = 0
    #     for i in range(0, width):
    #         Y_indices[i][j] = j # In Y_indices all columns will be:
    #         #[[0,1],
    #         #[0,1]]
    #         X_indices[i][j] = k # In X_indices all columns will be:
    #         #[[0,0],
    #         #[1,1]]
    #         k = k + 1
    # # Create new martix of ones in the same size-third channel will be ones matrix
    # ones_mat=np.ones((height,width))
    #
    # # Now we will create new 3D array where [X_indices,Y_indices,ones_mat] and shape is (rows,cols,3)
    # x_y_one_mat=[X_indices,Y_indices,ones_mat]
    # # Convert to np.array
    # x_y_one_mat=np.array(x_y_one_mat)
    # # Extract number of rows and cols
    # rows,cols=x_y_one_mat.shape[:2]
    # # Reshape it to (3,rows*cols) (page number 8)
    # x_y_one_mat_reshape=np.reshape(x_y_one_mat, (3, rows*cols))


#     # Display both image1 and the wrapped version of the image2 in the same figure
#     plt.imshow(im1)
#     plt.imshow(im2_wrapped)
#     return im2_wrapped # Return warp image 2 according to T
#
#     pass


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
    # https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
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

    gaussian = gaussianKer(5)  # Cefine the kernel
    for i in range(1, levels):
        # Expand the image in i+1 index from pyramid_list_gaussian i.e. expands a gaussian pyramid level one step up
        expand = gaussExpand(gauss_pyr[i], gaussian)
        lap_im = gauss_pyr[i - 1] - expand  # Create the difference image i.e. subtraction between gaussianPyr-x-Expand gaussianPyr x + 1
        pyramid_laplacian_list.append(lap_im)  # Add the new laplacian filtered image to the pyramid

    pyramid_laplacian_list.append(gauss_pyr[levels - 1])  # Add the last laplacian filtered image to the pyramid

    return pyramid_laplacian_list


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gaussian = gaussianKer(5)  # I did not use the same way as I defined the gaussian kernel before because it did not worked well.
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

    #  Now blend the two images with regard to the mask
    merge = (lap_pyr1[levels - 1] * gauss_pyr_mask[levels - 1]) + ((1 - gauss_pyr_mask[levels - 1]) * lap_pyr2[levels - 1])
    gaussian_kernel = gaussianKer(5)  # I did not use the same way as I defined the gaussian kernel before because it did not worked well.
    for i in range(levels - 2, -1, -1):
        merge = gaussExpand(merge, gaussian_kernel)
        merge = merge + (lap_pyr1[i] * gauss_pyr_mask[i]) + ((1 - gauss_pyr_mask[i]) * lap_pyr2[i])

    img_1 = cropSizeImage(img_1, levels)
    img_2 = cropSizeImage(img_2, levels)
    # Naive blending
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
    gs_k = (gs_k / gs_k.sum()) * 4  # As we learned in class
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
    height, width = img.shape[:2]
    height = twoPowLevel * np.floor(height / twoPowLevel).astype(np.uint8)
    width = twoPowLevel * np.floor(width / twoPowLevel).astype(np.uint8)

    if img.ndim == 3:
        img = img[0:height, 0:width, :]
    else:
        img = img[0:height, 0:width]
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
