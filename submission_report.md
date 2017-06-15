# Vehicle Detection Project

[//]: # (Image References)
[image1]: ./examples/hog-32-test1.jpg
[image2]: ./examples/hog-64-test1.jpg
[image3]: ./examples/hog-128-test1.jpg
[image6]: ./examples/slide-128-7-test1.jpg
[image7]: ./examples/slide-128-8-test1.jpg
[image8]: ./examples/slide-64-7-test1.jpg
[image9]: ./examples/slide-64-8-test1.jpg
[image10]: ./examples/slide-32-14-test1.jpg
[image11]: ./examples/slide-32-16-test1.jpg
[image12]: ./examples/heatmap-test1.jpg
[image13]: ./examples/detected-test1.jpg
[image14]: ./examples/test1.jpg
[video1]: ./project_video.mp4
[video2]: ./challenge_video.mp4

## Content of the Submission an d Usage
This submission includes the following python files:
* run.py: provides video processing pipeline for lane and vehicle detection.
* lane_detection.py: contains the LaneDetector class that implements lane detection pipeline enhanced from my Advanced Detection project submission for better lane detection
* vehicle_detection.py: contains the VehicleDetector class that provides the vehicle detection training and detection pipelines
* camera.py: contains the Camera class
* utils.py: contains some utility functions
* lane_config.py: configuration for lane detection
* vehicle_config.py: configuration for vehicle detection

### Usage
To run the lane and vehicle detection for a video stream:
````
python run.py -m trained_model -t n_threads -i input_video -o output_video
````

## The Pipelines
### The Video Pipeline
The detection pipeline contains the following steps:

* Create instances of the Camera, the LaneDetector, the VehicleDetector, and a ThreadPoolExecutor with two threads
* Load the video stream and iterate through each video frame
* For each frame:
    * Undistort the image
    * Launch the LaneDetector and VehicleDetector on two threads from the ThreadPoolExecutor to run lane and vehicle detection in parallel
    * Wait for the detectors to complete the detection pipelines
    * The results from the detectors contains a lane overlay image and vehicle overlay image, these two images are superimposed with the undistorted image that is them return to the  video clip for composing the output video

### The Vehicle Detection Pipeline
The Vehicle Detection Pipeline is composed of the following steps:
* Image normalization: normalize the image's RGB pixel values to the range of 0 to 255
* Convert the image to the designated colorspace
* Scan the image with the configured slides. Each slide is defined by the width of sliding window and vertical scan range. The side of the sliding window in each slide can have different height and width. For each slide:
    * Crop the image to the vertical scan range
    * Scale the image so that each sliding windows is scaled to the size of the training image size (64X64 in our case)
    * Compute the HOG for the resulting image
    * Sliding the (64X64) window horizontally then vertically on the resulting image according to the sliding step specified. For each step:
        * Extract HOG within the window
        * Extract the bin special and histogram from the image within the window
        * Concatenate all the above festures into a row array
    * For all sliding windows features extracted from the above steps, divide them into batches and run the prediction concurrently on multiple threads.
        * Each thread will first normalize the features using sklearn.preprocessing.RobustScaler, RobustScaler was chosen as it may perform better tha the StandardScaler though this requires further experiments.
        * After normalizing the features, the samples are than fed into the classifier for prediction.
        * Finally the position of the detected windows in the sliding window list are returned
    * Return the matched predictions
* Compose the heatmap from the matched predictions
* Compute the labels fromthe heatmap
* Get vehicle bounding boxes from the label
* Draw the bounding box on an overlay image
* Return the overlap image

#### Acceleration for Vehicle Detection
I used joblib's parallel processing utility to run predictions for multiple sliding windows in parallel. The process is configured to use threading backend so that threads instead of processes is used. This is to avoid more expensive process creation. Furthermore, a single thread pool is used during the processing of an image. The code for launching the parallel prediction is as follow:

````
with Parallel(n_jobs=config.predict_cpus, backend="threading") as par:
    ....
    ones = np.concatenate(par(delayed(self._predict)(features, cpu) for cpu in range(config.predict_cpus)))
````

In the above code, the samples (windows) are divided into a number consequent chunks equals to the number of threads, and a chuck is given to a thread for processing. 

It is possible that parallel processing can be utalized in other part of the piupeline, like extracting features from the images for multiple windows concurrently. But in my preliminary experiments, the prediction part seems to be the most time consuming part of the pipeline, and performing parallel extraction yielded little improvements. This might be due to the Amdahl's law where, in our case even it could possibly be done in parallel at a lower level, whole image HOG and fixed-cost per-window color features computation will limit how much speed up parallel processing can achieve. However, this will be investigated further when time permits.

In my test, the acceleration was able to speed up the process from over 850 second to 130 seconds for 18 images.

#### Vehicle Detection Classifiers
I have experimented with two classifiers, SVC, and SGD from Scikit-Learn. However, the program can be extended easily to support use of different classifiers like convolutional neural network. And CNN is definitely something for me to experiment with when time permits.

##### SVC
In Scilit-Learn's implementations, SVC requires all samples to be feed in a single training while it will perform internal iterations for accuracy. While it remain a question to me whether a generator could be used in the process to avoid loading all training samples into the memory, it could not be used for incremental (on-line) training.

The SVC was tuned using sklearn.model_selection.GridSearchCV with the following parameters:
````
{
    'kernel': ('linear', 'rbf'),
    'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
````
The final parameters settle on rbf for the kernel, and C=3. However, C seemed fine between 1 to 6 in my experiments.

##### SGD
SGD, on the other hand, provides partial_fit function for incremental training. The training process goes through several epochs, and in each epoch, the entire set of training samples are shuffled first, and are divided into batches. Each batch of samples are loaded from files, and fed into the training pipeline. 35 epochs were used in my training processes, and epochs whose test accuracy was close to 98%.

In addition to supporting incremental learning, SGD also has a much better performance than SVC, and this makes it a much better choice than SVC, at least when using Scikit-learn.

The SGD uses hinge loss function, and was tuned using sklearn.model_selection.GridSearchCV with the following parameters:
````
{
    'alpha': [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001],
    'epsilon': [0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5],
    'power_t': [ 0.4, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
}
````
And the results varies among different tuning tests, but alpha = 0.0002, epsilon = 0.4, and power_t=0.75 with optimal learning rate seem fine. However, alpha seems fine also with values between 0.0002 and 0.0008, epsilon seems fine between 0.1 to 0.8, and power_t for 0.4 to 0.8. Though more experiments are required to confirm this.

#### Histogram of Oriented Gradients (HOG)
*VehicleDetector._extract_hog()* extracts HOG features from an image. It indirectly calls the hog function of skimage.feature.

Different colorspaces were explored including RGB, HSV, LUV, and YCrCb. It turned out that YCrCb, and HSV performs better than others in training losses and prediction accuracy. Between YCrCb, and HSV, YCrCb is slightly better than HSV as I had observed. But more in-deepth analysis and tuning of the training parameters might change the picture.

As observed in my experiments, the choice of 8 pixels per cell, 2 cells per block, and 12 orientations seem fine from testing accuracy's perspective which is 98.6% for the model used for in the submission.

The following diagram illustrates hog images at different scale of sliding windows, and thje cropped image height:

|        | Slides and their HOGs |
|:---------------------:|:------------------:|
| 128x90, Height: 352   | ![image3]  		 |
|                       |                    |
| 64x64, Height: 128    | ![image2]  		 |
|                       |                    |
| 32x32, Height: 64     | ![image1]  		 |

#### Bin Spatial and Color Histogram
My experiments also show better validation accuracy could be achieved by including color features using color bin and histogram. This might make sense for cars have distinguished color from the surroundings. However, how the trained model can be generalized for different colors of color remains a question to me.

#### Sliding Windows
How the detection pipeline perform sliding windows is specified in vehicle_config.slidings. Each slide is specified with the size of the sliding window, and the vertical sliding range. I allow sliding windows to have different height and width. The detection is performed under *VehicleDetector.detect()*. Four slides were used in this report:

 1. Width: 128, height: 90, top: 320, bottom: 672
 2. Width: 64, height: 64), top: 386, bottom: 514
 3. Width: 32, height: 32), top: 422, bottom: 486

|        |Sliding Windows and their HOGs |
|:-----------------:|:------------------:|
| 128x90       | ![image6]  		     |
| 128x90       | ![image7]               |
|              |                         |
| 64x64        | ![image8]  		     |
| 64x64        | ![image9]               |
|              |                         |
| 32x32        | ![image10]  		     |
| 32x32        | ![image11]  		     |

 #### Heatmap and Detection Bounding Box
 The results of the prediction is used to create the heapmap. For each detected sliding window, the corresponding heatmap pixel value is incremented by one.
 The final heatmap is then filtered by the heatmap threshold, and result heatmap is then passed to scipy.ndimage.measurements.labels to label the heatmap.
 Finally the produced labels are then used to compute the bounding box for each label.

 The final step of the detection pipeline is to draw the bounding boxes on an overlay image that will be used to compose the final video image.

 The following images shows the Heatmap, the labeled bounding boxes, and the final image

 ![image12]
 ![image13]
 ![image14]

### The Lane Detection Pipeline
The lane detection pipeline extended my Advanced Lane detection submission in the following areas:
1. Provide a logic to mask off lane alternation marks
2. Provide a logic to detect and reject abrupt changes in detected lanes between subsequent frames
3. Implemented a mechanism to reset lane detector in order to recover from 'bad' detections from previous frames.

## The Result
The result of lane and vehicle detection for project_video.mp4 is: [project_video.mp4](./outputs/project_video.mp4)
The result of lane detection and vehicle detection for challenge_video can be found here: [challenge_video.mp4](./outputs/challenge_video.mp4)

There are still few issues shown in the result videos which might indicate the followings:

1. 97.9% test accuracy is still not good enough.
2. There need a more robust way of dealing with prediction errors, heatmap and labeling alone may not be good enough.

### Performance Comparison of SGD and SVC
The following table summaries the performance difference between SGD and SVC. It has shown that SDG is way superior to SVC in terms of training speed, detection speed, and trained model size!

|           |  Training                    | Detection              | Test Accuracy | Model Size (K bytes) |
|:---------:|:----------------------------:|:----------------------:|:-------------:|:--------------------:|
| SGD       | 2,974 seconds for 35 epochs  |   33 seconds	        |   97.9%       |        244           |
| SVC       | 5,632 seconds                |  707 seconds           |   95.9%       |    796,448           |

Training was performed with 17,323 training samples and tested with 3,057 samples. I have added around 3,500 images to improve the training for detecting left/right passing vehicles, non-vehicles like road, tree, road side, and dark shadows.

The detection time in the above table was measured using 18 images, and with 6 concurrent threads. For SGD, 97.9% test accuracy was achieved at the 19th epoch.

The video processing performance was 0.6 frame/second for SGD with 6 concurrent threads.

### Discussion and Further Works
Due to time restriction, many areas that I plan to explored in near future include:
1. Experiment with CNN
2. Further parameter tuning for improving the accuracy
3. Improve the detection to handle dark and show regions as my trained model is still not yet handle these situation well.
4. Improve the vehicle detection by filtering out bounding boxes that are considered errorous. This includes bounding boxes popping up from no where or vanishes to no where, bounding boxes in relatively flat areas ... etc
5. Make the bounding boxes move along the vehicles more smoothly by restricting how fast a bounging box can change between subsequent frames. The above two may be accomplished to some extent by identifying overlapping boxes in subsequent frames, but many more situations need to be addressed for this to work.
6. The performance of the detection is around 1.6 seconds per frame. This is too slow for real-time use. Therefore, further performance improvement by increasing the parallelism, and expore use of GPU acceleration is required

