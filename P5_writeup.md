**Vehicle Detection Project**

In this project, a SVM classifier is developed to detect cars for self-driving purposes. The SVM classifier is trained with 17000+ car and none car image datasets using Histogram of Oriented Gradients (HOG) and other features. The trained SVM is applied on the image using a sliding window approach. When applied on a video stream, the detection heatmap is constructed as a weighted average of multiple images to remove false detections. In the end, this vehicle detection pipeline successfully works on a video stream collected by a self-driving car. 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[img1]: ./output_images/cars.png
[img2]: ./output_images/notcar.png
[img3]: ./output_images/pp7.png
[img4]: ./output_images/pp9.png
[img5]: ./output_images/pp10.png
[img6]: ./output_images/pp14.png
[img7]: ./output_images/project.png
[img8]: ./output_images/hog1.png
[img9]: ./output_images/hog2.png
[img10]: ./output_images/hog3.png
[img11]: ./output_images/hog4.png



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG feature of the image captures the gradient flow of the object. It represents a very distinctive signature of the car objects that we want to identify. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![cars][img1]
![notcars][img2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). After experimenting with different color spaces ('RGB', 'GRAY', 'YUV', 'LUV', and 'YCrCb'). It is observed that the 'YCrCb' color space gives the best validation accuracy. In the YCrCb colorspace, Y is the luminance information, Cb is the blue component, Cr is the red component. Some features of the training images in the 'YCrCb' color space is visualized as follows.

![hog1][img8]
![hog2][img9]
![hog3][img10]
![hog4][img11]

#### 2. Explain how you settled on your final choice of HOG parameters.

There are multiple colorspaces and different HOG parameters to tune. I tested mulitple combintions of these parameters, and decided to use the following paramters which gives the best validation result.

    ### Tweak these parameters and see how the results change.
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 10#9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 64    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training process is pretty much by trial and error. After exploring different colorspace and different HOG parameters, the highest testing accuracy I got is about 98% with the parameters listed above. (Note that more training data are added during the exploration phase, but the total numbers of cars and notcars are approximately the same)

```
total number of cars: 8803, notcars: 9083 
('Using:', 10, 'orientations', 8, 'pixels per cell and', 2, 'cells per block')
('Feature vector length:', 9144)
(32.3, 'Seconds to train SVC...')
('Test Accuracy of SVC = ', 0.9852)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The SVM classifier is able to make binary decision of car or not car. To identify the object in the image, a sliding window search algorithm is adopted to find the location of the object in the image.

The sliding window approach has four parameters to tune, i.e., '[ystart, ystop, scale, cells_per_step]'. The 'ystart' and 'ystop' parameters crop out most of the trees, sky, and front part of the car. Larger 'scale' and 'cells_per_step' adopted for closer places, because closer cars are much larger in the figure. Finer search is performed for the further away part of the image. In the end, the following three sets of parameters are adopted for the sliding window search.

    # assign different scale for different ranges
    # [ystart, ystop, scale, cells_per_step]
    scaletable = [[400, 500, 1.0, 1],
                  [400, 600, 2.0, 1], 
                  [500, 656, 2.0, 2]]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result (with correct prediction and less false positives). The detection pipeline is outlined as follows:

* Convert the image to YCrCb colorspace
* Extract 3-channel HOG features plus spatially binned color and histograms of color in the feature vector
* Apply sliding window search on the image with three different scales
* Generate a heatmap of the identified boxes
* Threshold the heatmap and identify labeled objects from the image
* Plot boxes around the identified cars

Some examples of the detection pipeline are given below:

![pipepline][img4]
![pipepline][img5]
![pipepline][img6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

This car detection pipeline is applied on a video stream from a camera on the Udacity self-driving car. The identified cars are visualized with bounding boxes. In addition, the detection heat map and lane detection results are overlaid on the video.

##### Final video (click on the image to view the youtube video)

[![project][img7]](https://youtu.be/RbZO0yhE3YY)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The car detection pipeline was tuned based on the test images. It works very well on these finite set of images. However, when it is directly applied on the video stream, lots of false positives and misdetections appear. To address this issue, the detection heat map is generated, which stores the heatmap of the last six images in a sequence. 

```
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
```

The actual heatmap is calculated as a weighted average of the past six heatmaps. Then this discounting factor and the heatmap threshold parameter are tunned to make sure that false detections are filtered out and new cars are detected as soon as possible. (Note that small discount factor can heavily filter out false positives, but introduces a significant lag in the actual detection.)

```
    def calHeatMap(self, newhp):
        """
        calculate weight heatmap using current and history
        """
        heatmap = newhp.copy()
        discount = 0.7
        scale = 1.0*discount
        N = len(self.heatmaps)
        Knorm = 1.0
        for i in range(N-1, -1, -1):
            heatmap = heatmap +  scale*self.heatmaps[i]
            Knorm = Knorm+scale
            scale = scale*discount
        return heatmap/Knorm

```

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, we have developed a SVM classifier with a sliding window based sampling approach to identify cars in the image. This detection pipeline can be applied in real-time to track cars from the front facing camera. However, there are still multiple places that can be improved.

* Linear SVM model is used in this project to distinguish cars and nonecar objects. We get a validation accuracy of 0.9852 on about 17000 datasets. This dataset can be easily augmented by adding more training data from the internet. Also, a nonlinear SVM model can be used with different kernel functions. These changes can potentially make the classifier more accurate.  

* We can also replace the SVM classifier with CNN models. CNN models allow us to have complex models with more flexibility. A very good option is to use [YOLO](https://pjreddie.com/darknet/yolo/) developed by the University of Washington.

