## Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car_not_car.png
[image2]: ./writeup_images/car_hog.png
[image3]: ./writeup_images/not_car_hog.png
[image4]: ./output_images/classifier_output/test1.jpg
[image5]: ./output_images/final_output/test1.jpg
[image6]: ./output_images/final_output/test2.jpg
[image7]: ./output_images/final_output/test3.jpg
[image8]: ./output_images/heatmap_output/test1.jpg
[image9]: ./output_images/heatmap_output/test2.jpg
[image10]: ./output_images/heatmap_output/test3.jpg
[image8]: ./output_images/heatmap_output/test1.jpg
[image9]: ./output_images/heatmap_output/test2.jpg
[image10]: ./output_images/heatmap_output/test3.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! Please find the intermediate output images for all test images [here](https://github.com/AkshathaHolla91/CarND-Vehicle-Detection/tree/master/output_images)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the sixth code cell of the IPython notebook `vehicle_detection.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I have used all channels in the YCrCb colorspace while calculating hog features as it gives a better accuracy compared to individual channels.

Here is an example using  HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, `histogram_bins=8` ,`spatial_size=(16,16)`:


![alt text][image2]
![alt text][image3]



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and initially started with RGB colorspace and then other colorspaces like HLS and HSV, in the end I settled with the YCrCb colorspace as I found that it gave me the best results when I used each of the colorspaces separately while running my classifier.

When trying out different values with orientations I found that lower values of orientations were not succesful in detecting vehicles as compared to higher ones and since the suggested range is in between 6 and 12 I tried the higher values and found 9 orientations to be optimal.

I also found that keeping the pixels_per_cell at (8,8) and cells per block at (2,2) gives a good accuracy with the classifier that I am using.

In case of spatial binning dimensions and histogram bins  I started with the default (32, 32)  for spatial dimensiona and went down till (8,8) and found that the (16, 16 ) was optimal, similarly 8 histogram bins worked well with my linear classifier and produced good results.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Here I have used a Linear support vector machine classifier(sklearn's LinearSVC) as seen in code cell 12 of vehicle_detection.ipynb. Here I have extracted the features of  the vehicle  and non vehicle data sets using the extract features function(cell 6) as mentioned above and then prepared the features(X) by stacking the car and not car features obtained and also the labels by setting ones for the length of car features and zero for non car features,  later I used the train_test_split function to generate the training and test data set by dividing the sample data into 80% training set and the rest 20% as the test set after shuffling the samples.I then used the StandardScaler  function to fit the training feature set and using the obtained scaler transformed the training feature set and the test feature set. I used the resulting X_train(training features) and Y_train(training labels) to train my classifier which gives accuracy in the range of 98 to 99%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have implemented the hog sub sampling window search in code cell 15 of my jupyter notebook in the find_cars_with_heatmap function. Here I am passing the ystart and ystop positions ie  the  region of the image over which the search for vehicle has to be done(region of image having lane , in this case 400 to 656 along y) and also a list of the scales. Here I have chosen 4 scales in the range of 1 to 2.5 with a difference of 0.5 to make sure that cars are detected at most scales when the search is happening. Here I have chosen a window size of 64 8x8 cells with 8 pixels per cell and 2 cells per block. This method extracts the hog features once, for each of a small set of window sizes , and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor that impacts the window size which changes according to the provided list. Here the overlap of each window can be defined in terms of cells per step  which I have taken as 2. The hog features , spatial features and hist features are later calculated for each identified window and  fed to a classifer to make a prediction  to identify whether a car/ vehicle has been identified. If it is found a box is drawn according to the calculated positions in the window and is repeated for all windows and all steps as seen in the code until a list of boxes identifying the cars in the image are obtained.

The image below shows the result of the hog sub sampling search applied on a test image. Here we can see multiple predictions for the cars which were calculated in various windows.

![alt text][image4]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on the four  scales mentioned above using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is the pipeline result for one of the test images:

![alt text][image7]




Initially I used heatmaps to get the hot areas (areas with multiple positive prediction values for cars) to generate a single bounding box to identify each car instead of the multiple bounding boxes generated earlier. I have also maintained the information of identified bounding boxes of the previous 8 frames to maintain consistency in detection of vehicles (smoothening of detection) when atleast 2 bounding boxes are identified in the region(ie with a threshold of 1) and given it as the input to the heatmap thresholding and label function to properly detect the vehicle throughout the frames.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/AkshathaHolla91/CarND-Vehicle-Detection/blob/master/final_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is an example of the generated  heatmaps for the test image above

![alt text][image8]


### Here the resulting bounding boxes are drawn on the test image
![alt text][image5]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

While building the pipeline I initially tried training my classifier while using the RGB colorspace which led to detection of multiple false positives because of which I had to change the colorspace which eventually led me to use a different colorspace which was YCrCb. I also tried using the sliding window approach for searching the image and making the predictions which turned out to be slow and inefficient. I later changed that to hog sub sampling search which was faster and more efficient. After training my linear SVM with the entire data set , the video result that I got still had some inconsistencies in terms of maintaining the shape of the bounding box which was fluctuating a lot. I later fixed this by maintaining the values of detected boxes of the previous 8 frames and passing it to the heatmap function to maintain proper continuity and smoothness in detection of the vehicle.  It also took a lot of time to create the final output video due to limited hardware capability since I am using a pentium processor.

Currently the pipeline is able to detect most vehicles in its surroundings with minimal false positives. 

The area where it might fail would be when the dimensions and build of the cars or the surrounding environment and lanes  of the car were to be drastically different from the training set.

One improvement would be to use other classifiers on the existing data set instead of the existing Linear SVM.



Using Deep learning approach to this problem could also bring a possible improvement over the existing accuracy and detection.

Using the YOLO model for object detection would be another possible improvement in this regard.

