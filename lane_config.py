'''
Configuration for Advanced Lane Detection
'''
import os
import numpy as np

class LaneConfig:
    '''
    Config class
    '''
    def __init__(self):
        self.crop = None
        self.trapezoid_y = None
        self.trapezoid_x = None
        self.scan_thresh = None

    def __exit__(self, *args):
        pass

    def __enter__(self, *args):
        return self

    def set(self, name=None):
        '''
        Set configuration
        name: name of the configuration
        '''
        trapezoids = None
        if name is not None:
            name = os.path.splitext(os.path.basename(name))[0]
            for key in self.trapezoid:
                if name.find(key) == 0:
                    print('Use configuration: ', key)
                    trapezoids = self.trapezoid[key]
                    break
        if trapezoids is None:
            print('Use default configuration')
            trapezoids = self.trapezoid['default']
        self.crop = trapezoids[0]
        self.trapezoid_y = trapezoids[0]
        self.trapezoid_x = trapezoids[1]
        self.scan_thresh = int(self.sliding_width * self.layer_height * self.scan_thresh_ratio)

config = LaneConfig()

# Test flag, True to allow more data to be made available for test
config.test = False
# Use best fit to fit lanes
config.bestfit = True
# Use best fit to fit lanes
config.lane_shift_thresh = 75
# True to save video images
config.save_video_images = False
# Parallel trapezoids for computing parallel-perspective projection transformation for different videos
config.trapezoid = {
    'default': [[470, 690], [260, 565, 720, 1060]],
    'challenge_video': [[490, 690], [332, 590, 740, 1080]],
    'harder_challenge_video': [[500, 675], [250, 500, 737, 963]]
}
# Folder containing the camera calibration images
config.calibration_folder = "camera_cal/"
# Folder containing the test images
config.test_image_folder = "test_images/"
# Size of the camera images
config.image_size = (720, 1280)
# Chessboard cells' shape
config.chessboard_shape = (9,6)
# Height of a scan layer
config.layer_height = 60
# When we start scan an image for the initial left and right points, we divide the image
# height by the start_divide, then we search the initial point by looping from start_divide to 2.
# In each loop, the bottom 1/start_divide portion of the image is search
# The loop terminates until we find a starting point or we reach half of the image
config.start_divide = 6
# The number of vertical layers on the transformed image for used for lane detection
config.scan_layers = 12
# Height of a scan layer
config.layer_height = 60
# Width of the sliding windows in each scan
config.sliding_width = 50
# Width of the scan range to the left and right of the lane detected in the layer below
config.scan_width = 75
# When scan for lane line, we increase the window scan window after a miss until we reach the max_scan_width
config.max_scan_width = 150
# The number of on pixels in a sliding window that need to be on for the window to be considered 
config.scan_thresh_ratio = 0.125
# The lane width variation threshold
config.width_variation = 0.15
# The maximal number of detection failures to trigger a reset
config.failure_reset_thresh = 3
# The number of on pixels in a sliding window that need to be on for the window to be considered 
config.scan_thresh_ratio = 0.125
# Lane smooth factor, p = p * smooth_factor + (*1 - smooth_factor) * p1
config.smooth_factor = 0.4
# Curverature change threshold between subsequent lines
config.curverature_threshold = 100
# Sobel detection kernel
config.sobel_kernel = 3
# Sobel threshold
config.sobel_thresh = (20, 100)
# Sobel magnitude threshold 
config.magnitude_thresh = (80, 100)
# HSL colorspace satuation threshold
config.hls_thresh = None
# HSV threshold to filter image by white and yellow colors. The first range is for pure white,
# the second is for near white which can be any color with low saturation, and the third is for yellow
config.hsv_thresh = [(np.uint8([0, 0, 215]), np.uint8([180, 30, 255])),
                     (np.uint8([14, 80, 100]), np.uint8([30, 255, 255]))]
# Lane change mark colors
config.hsv_maskoff = [(np.uint8([0, 0, 0]), np.uint8([180, 50, 150]))]

config.set()
