# lane detection pipeline video
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
# convert float image to uint8 image
from skimage import img_as_ubyte

# modules from lane detection project
from lane_detection_pipeline.pp_transform import corners_unwarp  
from lane_detection_pipeline.lane_detection import evalPoly, thresholdIMG, findLanes, visualLane

# modules for car detection
from detection_validation_pipeline import find_cars, filterBox, getHeatmap


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
    def __init__(self, clipname, mtx, dist, svc_pickle):
        # prepare input/output filenames
        self.clip = VideoFileClip(clipname)
        out_dir='output_video/processed_'
        self.output = out_dir+clipname
        # camera calibration parameters
        self.mtx = mtx
        self.dist = dist
        # trained svc model
        self.svc = svc_pickle
        # left and right lane object
        self.lL = lane()
        self.lR = lane()
        # build heat map history
        self.heatmaps = []

    def updateHeatMap(self, newhp):
        """
        update heatmap (thresholded), store in memory
        """
        # averaging over past Nmax heatmaps
        Nmax = 6
        N = len(self.heatmaps)
        if N<Nmax:
            self.heatmaps.append(newhp)
        else:
            self.heatmaps.append(newhp)
            self.heatmaps.pop(0)
        return

    def calHeatMap(self, newhp):
        """
        calculate weight heatmap using current and history
        """
        heatmap = newhp.copy()
        discount = 0.6
        scale = 1.0*discount
        N = len(self.heatmaps)
        Knorm = 1.0
        for i in range(N-1, -1, -1):
            heatmap = heatmap +  scale*self.heatmaps[i]
            Knorm = Knorm+scale
            scale = scale*discount
        return heatmap/Knorm

        

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


        # overlay car detection on lane detection result
        # load attributes of our svc object
        svc = self.svc["svc"]
        X_scaler = self.svc["scaler"]
        orient = self.svc["orient"]
        pix_per_cell = self.svc["pix_per_cell"]
        cell_per_block = self.svc["cell_per_block"]
        spatial_size = self.svc["spatial_size"]
        hist_bins = self.svc["hist_bins"]
        
        #result = img.copy()

        # find cars with sliding window    
        out_img, box_list = find_cars(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        # get raw heatmap
        heatmap_raw = getHeatmap(result, box_list)
        heatmap_weighted = self.calHeatMap(heatmap_raw)

        # filter car positions and output heat map    
        draw_img, heatmap = filterBox(result, heatmap_weighted)

        self.updateHeatMap(heatmap)
        return draw_img


    def process_video(self):
        """
        feed individual images into the image processor and generate output video
        """
        out_clip = self.clip.fl_image(self.process_image_showlane)
        out_clip.write_videofile(self.output, audio=False)




if __name__ == "__main__":
    # Read in the saved camera matrix and distortion coefficients
    lane_pickle = pickle.load( open( "lane_detection_pipeline/calibration_undistort.p", "rb" ) )
    mtx = lane_pickle["mtx"]
    dist = lane_pickle["dist"]

    # load a pe-trained svc model from a serialized (pickle) file
    modelname = "Trained_model/2018-03-19-trained_SVM.p"
    svc_pickle = pickle.load( open(modelname, "rb" ) )

    # construct a video processor with calibration data
    #vp = videoProcessor("test_video.mp4", mtx, dist, svc_pickle)
    vp = videoProcessor("project_video.mp4", mtx, dist, svc_pickle)
    #vp = videoProcessor("p9.mp4", mtx, dist, svc_pickle)


    # start processing video and generate output video
    vp.process_video()
