# perspective transformation
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#unwarp the image
def corners_unwarp(img, mtx, dist):
    # Undistort using mtx and dist
    img_undistort = cv2.undistort(img, mtx, dist, None, mtx)
    imgsize = (img.shape[1], img.shape[0]) # (maxwidth, maxheight)
    # pick up 4 src points
    # line1, short horizon
    #src = np.float32([[264., 670.],[576., 460.],[705., 460.],[1042., 670.]])
    # line2, long horizon
    #src = np.float32([[272., 673.],[620., 430.],[661., 430.],[1052., 673.]])
    # line2, moderate horizon
    src = np.float32([[272., 673.],[593., 450.],[691., 450.],[1052., 673.]])
    # set up 4 target points (assume flat ground, 1280, 720)
    dst = np.float32([[300., 720.],[300, 0],[980, 0],[980., 720.]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_undistort, M, imgsize, flags=cv2.INTER_LINEAR)
    return warped, M

if __name__ == "__main__":
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open( "camera_cal/calibration_undistort.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Read in an image
    img = cv2.imread('test_images/straight_lines2_src.jpg')
    # calculate perspective transform
    top_down, perspective_M = corners_unwarp(img, mtx, dist)


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted and Warped Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
