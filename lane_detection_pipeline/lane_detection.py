# lane detection pipeline video
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# convert float image to uint8 image
from skimage import img_as_ubyte
from pp_transform import corners_unwarp  


def evalPoly(fit_param, Y):
    """
    Evaluate X, based on Y of the polynomial
    """
    return fit_param[0]*Y**2 + fit_param[1]*Y + fit_param[2]

def thresholdIMG(img, sx_thresh=(40, 255), l_thresh = (220, 255), b_thresh = (155,255)):
    """
    Thresholding original image with 3 different criteria
    """
    # Convert to HLS color space and use the L channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel>=l_thresh[0])&(l_channel<=l_thresh[1])] = 1
    # calculate gradient in x direction
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Convert to LAB color space, and use the B channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    b_channel = lab[:,:,2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel>=b_thresh[0])&(b_channel<=b_thresh[1])] = 1
    # combine detection binaries
    img_out = np.dstack(( sxbinary, l_binary, b_binary))
    return img_out


def findLanes(top_down):
    """
    extract lanes from top_down view of the road
    """
    binary_warped = np.zeros((top_down.shape[0], top_down.shape[1]))
    binary_warped[(top_down[:,:,0]>0) | (top_down[:,:,1]>0) | (top_down[:,:,2]>0)] = 1

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = np.uint8(out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    pts_raw = [leftx, lefty, rightx, righty]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, top_down.shape[0]-1, top_down.shape[0] )
    left_fitx = evalPoly(left_fit, ploty)
    right_fitx = evalPoly(right_fit, ploty)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))

    return pts, pts_raw, out_img

def visualLane(image, pts, pts_raw, perspective_M):
    """
    Visualize the detected lane, radius, and car center shift
    """
    # plot on original image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, pts, (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_zero, np.linalg.inv(perspective_M), (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    ymax = float(image.shape[0])
    y_eval = ymax
    leftx = pts_raw[0]
    lefty = pts_raw[1]
    rightx = pts_raw[2]
    righty = pts_raw[3]
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    # print distance from center and radius on the image
    lane_center = (evalPoly(left_fit_cr, ymax*ym_per_pix) + evalPoly(right_fit_cr, ymax*ym_per_pix))/2.0
    car_center = image.shape[1]*xm_per_pix/2.0
    str1 = "Distance from center: {:2.2f} m".format(car_center-lane_center)
    str2 = "Radius of Curvature: {:2.2f} km".format((left_curverad+right_curverad)/2000.)
    cv2.putText(result,str1,(430,630), cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255))  
    cv2.putText(result,str2,(430,660), cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255))    
    return result



if __name__ == "__main__":
    # load camera calibration data
    dist_pickle = pickle.load( open( "camera_cal/calibration_undistort.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # load image
    #image = mpimg.imread('test_images/p3.png')
    image = mpimg.imread('test_images/test6.jpg')
    image = img_as_ubyte(image)
    # threshold image
    img_thresh = thresholdIMG(image)
    # unwarp image
    top_down, perspective_M = corners_unwarp(img_thresh, mtx, dist)

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(img_thresh)
    #plt.save_figure(result, "result.jpg")
    ax2.set_title('Threshold Result', fontsize=20)
    ax3.imshow(top_down)
    #plt.save_figure(result, "result.jpg")
    ax3.set_title('Pipeline Result', fontsize=20)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


    fig = plt.figure(2)
    # find lane line pixels
    pts, pts_raw, out_img = findLanes(top_down)
    plt.imshow(np.uint8(out_img))
    N = pts.shape[1]
    plt.plot(pts[0, 0:N/2, 0], pts[0, 0:N/2, 1], color='yellow')
    plt.plot(pts[0, N/2:, 0], pts[0, N/2:, 1], color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


    plt.figure(3)
    result = visualLane(image, pts, pts_raw, perspective_M)
    plt.imshow(result)
    plt.show()

    print("finish plotting")
