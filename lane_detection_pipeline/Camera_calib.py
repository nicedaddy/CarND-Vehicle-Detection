# camera calibration pipeline
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

def detectCorner(nx, ny, images):
    """
    detect corners in the calibration images
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            write_name = 'camera_cal/detected/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            print("finished figure %s" %idx)
    return objpoints, imgpoints



if __name__ == "__main__":
    # specify corners in the calibration image
    nx, ny = 9, 6

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # detect corners in calibration images
    objpoints, imgpoints = detectCorner(nx, ny, images)

    # Test undistortion on an image
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    print("Calculating camera matrix")
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    print("Undistorting ...")
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('camera_cal/undistorted.jpg',dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/calibration_undistort.p", "wb" ) )
    print("saved to pickle file")

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()



