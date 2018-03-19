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


The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![cars][img1]
![notcars][img2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


YCbCr
In this format, luminance information is stored as a single component (Y), and chrominance information is stored as two color-difference components (Cb and Cr). Cb represents the difference between the blue component and a reference value. Cr represents the difference between the red component and a reference value. (YUV, another color space widely used for digital video, is very similar to YCbCr but not identical.)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

    ### Tweak these parameters and see how the results change.
    color_space = 'YCrCb'#'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
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


![hog1][img8]
![hog2][img9]
![hog3][img10]
![hog4][img11]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

```
total number of cars: 8803, notcars: 9083 
('Using:', 10, 'orientations', 8, 'pixels per cell and', 2, 'cells per block')
('Feature vector length:', 9144)
(32.3, 'Seconds to train SVC...')
('Test Accuracy of SVC = ', 0.9852)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

    # assign different scale for different ranges
    # [ystart, ystop, scale, cells_per_step]
    scaletable = [[400, 500, 1.0, 1],
                  [400, 600, 2.0, 1], 
                  [500, 656, 2.0, 2]]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![pipepline][img4]
![pipepline][img5]
![pipepline][img6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Final video (click on the image to view the youtube video)

This car detection pipeline is applied on a video stream from a camera on the Udacity self-driving car. The identified cars are visualized with bounding boxes. In addition, the detection heat map and lane detection results are overlaid on the video.

[![project][img7]](https://youtu.be/RbZO0yhE3YY)


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

* We can also replace the SVM classifier with DNN models. DNN models allow us to have complex models with more flexibility. A very good option is to use [YOLO](https://pjreddie.com/darknet/yolo/) developed by the University of Washington.

