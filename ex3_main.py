from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):  # work gray & RGB
    print("LK Demo")

    # img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, (img_1.shape[1], img_1.shape[0]))

    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):  # work gray &RGB
    """
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

    ans = opticalFlowPyrLK(im1.astype(np.float), im2.astype(np.float), 4, 20, 5)

    x_y = np.array([])
    u_v = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                u_v = np.append(u_v, ans[i][j][0])
                u_v = np.append(u_v, ans[i][j][1])
                x_y = np.append(x_y, j)
                x_y = np.append(x_y, i)
    x_y = x_y.reshape(int(x_y.shape[0] / 2), 2)
    u_v = u_v.reshape(int(u_v.shape[0] / 2), 2)
    print(np.median(u_v, 0))
    print(np.mean(u_v, 0))
    displayOpticalFlow(im2, x_y, u_v)


def compareLK(img_path):  # work gray&RGB
    """
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")

    # im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=np.float)
    height, width = im1.shape[:2]

    im2 = cv2.warpPerspective(im1, t, (width, height))

    x_y, u_v = opticalFlow(im1.astype(np.float), im2.astype(np.float), 20, 5)
    ans = opticalFlowPyrLK(im1.astype(np.float), im2.astype(np.float), 4, 20, 5)
    x_y_pyr = np.array([])
    u_v_pyr = np.array([])

    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                u_v_pyr = np.append(u_v_pyr, ans[i][j][0])
                u_v_pyr = np.append(u_v_pyr, ans[i][j][1])

                x_y_pyr = np.append(x_y_pyr, j)
                x_y_pyr = np.append(x_y_pyr, i)

    x_y_pyr = x_y_pyr.reshape(int(x_y_pyr.shape[0] / 2), 2)
    u_v_pyr = u_v_pyr.reshape(int(u_v_pyr.shape[0] / 2), 2)

    f, ax = plt.subplots(1, 3)
    ax[0].set_title('LK')
    ax[0].imshow(im2, cmap="gray")
    ax[0].quiver(x_y[:, 0], x_y[:, 1], u_v[:, 0], u_v[:, 1], color='r')

    ax[1].set_title('Pyramid LK')
    ax[1].imshow(im2, cmap="gray")
    ax[1].quiver(x_y_pyr[:, 0], x_y_pyr[:, 1], u_v_pyr[:, 0], u_v_pyr[:, 1], color='r')

    ax[2].set_title('overlapping')
    ax[2].imshow(im2, cmap="gray")
    ax[2].quiver(x_y[:, 0], x_y[:, 1], u_v[:, 0], u_v[:, 1], color='r')
    ax[2].quiver(x_y_pyr[:, 0], x_y_pyr[:, 1], u_v_pyr[:, 0], u_v_pyr[:, 1], color='y')
    plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
def imageWarpingDemo(img_path):
    """
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")
    img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -8],
                  [0, 1, -5],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    start_time = time.time()
    # warpImages(img1.astype(float), img2.astype(float), t)
    end_time = time.time()
    print("Time: ", (end_time - start_time))

    t = np.array([[np.cos(np.deg2rad(5)), -np.sin(np.deg2rad(5)), 10],
                  [np.sin(np.deg2rad(5)), np.cos(np.deg2rad(5)), 10],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    # warpImages(img1.astype(float), img2.astype(float), t)

    t = np.array([[np.cos(np.deg2rad(10)), -np.sin(np.deg2rad(10)), 0],
                  [np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10)), -10],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    warpImages(img1.astype(float), img2.astype(float), t)


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):  # work gray&RGB
    print("Gaussian Pyramid Demo")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255

    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    if img.ndim == 3:  # In case of RGB
        canvas = np.zeros((canv_h, canv_w, 3))
    else:  # In case of BW
        canvas = np.zeros((canv_h, canv_w))
    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        if img.ndim == 3:
            canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

        else:
            canvas[:h, widths[lv_idx]:widths[lv_idx + 1]] = gau_pyr[lv_idx]

    if img.ndim == 3:
        plt.imshow(canvas)
        plt.show()
    else:
        plt.imshow(canvas, cmap="Greys_r")
        plt.show()


def pyrLaplacianDemo(img_path):  # work gray&RGB
    print("Laplacian Pyramid Demo")

    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():  # work gray&RGB
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# --------------------- My Additional Tests ---------------------
# ---------------------------------------------------------------------------

def translationlkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Translation LK Demo")
    img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -2],
                  [0, 1, -1],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])

    start_time = time.time()
    ans = findTranslationLK(img1.astype(float), img2.astype(float))
    end_time = time.time()

    print("Time:", (end_time - start_time))
    print("MSE between both images: " + str(np.square(img2 - img1).mean()))
    print("MSE second img to the returned img: " + str(np.square(img2 - cv2.warpPerspective(img1, ans, img1.shape[::-1])).mean()))
    print("U,V Original: " + str("-2,-1"))
    print("U,V I get: " + str(ans[0][2]) + ", " + str(ans[1][2]) + "\n")
    #now plot image
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img1)
    ax[0].set_title("Image 1")
    ax[2].imshow(img2)
    ax[2].set_title("Image 2")
    ax[1].imshow(cv2.warpPerspective(img1, ans, img1.shape[::-1]))
    ax[1].set_title("Translated image")
    plt.show()

def translationCorrDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Translation Correlation Demo")
    img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -2],
                  [0, 1, -1],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])

    start_time = time.time()
    ans = findTranslationCorr(img1.astype(float), img2.astype(float))
    end_time = time.time()

    print("Time:", (end_time - start_time))
    print("MSE between both images: " + str(np.square(img2 - img1).mean()))
    print("MSE second img to the returned img: " + str(np.square(img2 - cv2.warpPerspective(img1, ans, img1.shape[::-1])).mean()))  # (mean_squared_error(img2, cv2.warpPerspective(img1, ret, img1.shape[::-1]))))
    print("U,V Original: " + str("-2,-1"))
    print("U,V I get: " + str(ans[0][2]) + ", " + str(ans[1][2]) + "\n")


def rigidlkdemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Rigid LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    t1 = np.array([[1, 0, -.7],
                   [0, 1, 0.4],
                   [0, 0, 1]], dtype=np.float)
    t2 = np.array([[np.cos(0.05), -np.sin(0.05), 0],
                   [np.sin(0.05), np.cos(0.05), 0],
                   [0, 0, 1]], dtype=np.float)

    t = t1 @ t2
    img_2 = cv2.warpPerspective(img_1, t, (img_1.shape[1], img_1.shape[0]))
    start_time = time.time()
    rigid_mat = findRigidLK(img_1, img_2)
    et = time.time()
    print("Time: ", (et - start_time))
    print("mat\n", rigid_mat, "\nt\n", t)

    new = cv2.warpPerspective(img_1, rigid_mat, (img_1.shape[1], img_1.shape[0]))
    if len(img_2.shape) == 2:
        f, ax = plt.subplots(1, 3)
        ax[0].set_title('img2 given transformation')
        ax[0].imshow(img_2, cmap='gray')

        ax[1].set_title('img2 found transformation')
        ax[1].imshow(new, cmap='gray')

        ax[2].set_title('diff')
        ax[2].imshow(img_2 - new, cmap='gray')

        plt.show()
    print("mse= ", np.square(new - img_2).mean())


def rigidCorrdemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Rigid Corr Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # I checked with another values but with those, the MSE was minimum
    t1 = np.array([[1, 0, -0.7],
                   [0, 1, 0.4],
                   [0, 0, 1]], dtype=np.float)
    theta = 0.05
    t2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]], dtype=np.float)

    t = t1 @ t2
    img_2 = cv2.warpPerspective(img_1, t, (img_1.shape[1], img_1.shape[0]))
    start_time = time.time()
    rigid_mat = findRigidCorr(img_1, img_2)  # Only thing I change
    et = time.time()
    print("Time: ", (et - start_time))
    print("mat\n", rigid_mat, "\nt\n", t)

    new = cv2.warpPerspective(img_1, rigid_mat, (img_1.shape[1], img_1.shape[0]))
    if len(img_2.shape) == 2:
        f, ax = plt.subplots(1, 3)
        ax[0].set_title('img2 given transformation')
        ax[0].imshow(img_2, cmap='gray')

        ax[1].set_title('img2 found transformation')
        ax[1].imshow(new, cmap='gray')

        ax[2].set_title('diff')
        ax[2].imshow(img_2 - new, cmap='gray')

        plt.show()
    print("mse= ", np.square(new - img_2).mean())


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    # lkDemo(img_path) #work good Gray&RGB
    # hierarchicalkDemo(img_path) #work good Gray&RGB
    # compareLK(img_path)#work good Gray&RGB
    #imageWarpingDemo(img_path)
    # pyrGaussianDemo('input/pyr_bit.jpg')#work good Gray&RGB
    # pyrLaplacianDemo('input/pyr_bit.jpg')
    # blendDemo()

    translationlkDemo(img_path) #work good translation
    # rigidlkdemo(img_path) #work good rigid lk
    # rigidCorrdemo(img_path) #work good
    # translationCorrDemo(img_path) #work good
###***Credit to photographer Daniel Appel

if __name__ == '__main__':
    main()
