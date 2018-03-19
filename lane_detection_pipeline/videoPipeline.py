# lane detection pipeline video
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
# convert float image to uint8 image
from skimage import img_as_ubyte

# modules built along the way
from pp_transform import corners_unwarp  
from lane_detection import evalPoly, thresholdIMG, findLanes, visualLane

class lane:
    """
    lane class to store lane parameters from frame to frame
    """
    def __init__(self):
        # initialize lane parameters
        self.left = []
        self.right = []
        # lane radius of curvature and car shift w.r.t. lane center
        self.R = 0
        self.shift = 0

    def updateLane(self, left, right):
        # update new left and right lane parameters
        self.left = left
        self.right = right

class videoProcessor:
    """
    video processing to find lanes
    """
    def __init__(self, clipname, mtx, dist):
        # prepare input/output filenames
        self.clip = VideoFileClip(clipname)
        out_dir='output_video/processed_'
        self.output = out_dir+clipname
        # camera calibration parameters
        self.mtx = mtx
        self.dist = dist
        # left and right lane object
        self.lL = lane()
        self.lR = lane()
    
    def process_image(self, img):
        """
        process a single image, and find lane
        """
        # convert float image to uint8, if image is loaded as float
        # img = img_as_ubyte(img)
        # threshold image
        img_thresh = thresholdIMG(img)
        # unwarp the image and get perspective transformation matrix
        top_down, perspective_M = corners_unwarp(img_thresh, self.mtx, self.dist)
        # find lane line pixels
        pts, pts_raw, out_img = findLanes(top_down)
        # update lane representation, this infomation can be used "hot" start the next sliding window
        # self.lL = pts_raw[0]
        # self.lR = pts_raw[2]
        # visualize lane and parameters
        result = visualLane(img, pts, pts_raw, perspective_M)
        return result

    def process_image_showlane(self, img):
        """
        process a single image, find lane, overlay the sliding window on top
        """
        # convert float image to uint8, if image is loaded as float
        # img = img_as_ubyte(img)
        # threshold image
        img_thresh = thresholdIMG(img)
        # unwarp the image and get perspective transformation matrix
        top_down, perspective_M = corners_unwarp(img_thresh, self.mtx, self.dist)
        # find lane line pixels
        pts, pts_raw, out_img = findLanes(top_down)
        # update lane representation, this infomation can be used "hot" start the next sliding window
        # self.lL = pts_raw[0]
        # self.lR = pts_raw[2]
        # visualize lane and parameters
        result = visualLane(img, pts, pts_raw, perspective_M)
        # overlay lane detection on the original image
        # x 420 to 840, y 0 to 240
        resized_img = cv2.resize(out_img, (420, 240)) 
        for c in range(0, 3):
            # overlay detection video with 0.5 transparentcy
            result[0:240, 420:840, c] = (0.5*result[0:240, 420:840, c] + 0.5*resized_img[:, :, c])
        return result


    def process_video(self):
        """
        feed individual images into the image processor and generate output video
        """
        out_clip = self.clip.fl_image(self.process_image_showlane)
        out_clip.write_videofile(self.output, audio=False)


if __name__ == "__main__":
    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load( open( "camera_cal/calibration_undistort.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # construct a video processor with calibration data
    #vp = videoProcessor("small.mp4", mtx, dist)
    vp = videoProcessor("project_video.mp4", mtx, dist)
    #vp = videoProcessor("challenge_video.mp4", mtx, dist)
    #vp = videoProcessor("harder_challenge_video.mp4", mtx, dist)

    # start processing video and generate output video
    vp.process_video()

