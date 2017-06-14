'''
Lane detetion
'''
import math
import numpy as np
import cv2
import utils

class Lane():
    def __init__(self, config, min_y, max_y, smooth_factor=0.5, distance_thresh=75, curverature_threshold=100):
        # Minimal, and maximal y coordinate
        self.config = config
        self.min_y = min_y
        self.max_y = max_y
        self.middle = (min_y + max_y) / 2
        # Line smooth factor for interpolation between previous fit and new fit
        self.smooth_factor = smooth_factor
        # Max distance between subsequent updates
        self.distance_thresh = distance_thresh
        # Max curverature different between subsequent updates
        self.curverature_threshold = curverature_threshold
        # was the line detected in the last iteration?
        self.detected = False
        # Number of consecutive fail detections
        self.fails = 0
        # x values of the last n detections
        self.recent_detects = []
        # x values of the last n fits of the line
        self.recent_fits = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        # The last detected lane points
        self.current = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        self.history_size = 4
        self.committed = True

    def set(self, points):
        '''
        Set the fit's coefficient
        '''
        if points is not None and len(points) > 1: # we need at least two points
            w = np.ones((len(points), ), np.float)
            w[1] = 2.
            if len(points) > 3: # minimal 4 points for second order polynomial
                w[2] = 1.5
                fit = np.polyfit(points[:, 1], points[:, 0], 2)
            else:
                fit = np.insert(np.polyfit(points[:, 1], points[:, 0], 1), 0, 0.)

            if self.current_fit is not None:
                dist = self.dist(fit)
                if dist[1] > self.distance_thresh: # reject the line as it has exceeded the max change threshold
                    if self.config.test:
                        print("Change in distance too big: ", dist)
                    self.detected = False
                    self.fails += 1
                    return False
                else:
                    radius_of_curvature = self.curverature(fit, self.middle)
                    diff = abs(radius_of_curvature - self.radius_of_curvature)
                    if 2*diff/(radius_of_curvature+self.radius_of_curvature) > self.curverature_threshold:
                        if self.config.test:
                            print("Change in curverature too big: ", radius_of_curvature, self.radius_of_curvature, 
                                  radius_of_curvature / diff)
                        self.detected = False
                        self.fails += 1
                        return False
                    self.radius_of_curvature = radius_of_curvature
                    self.current_fit = fit
            else:
                self.current_fit = fit
                self.radius_of_curvature = self.curverature(self.current_fit, self.middle)

            self.current = points
            self.recent_detects.append(points)
            if len(self.recent_detects) > self.history_size:
                self.recent_detects.pop(0)

            self.recent_fits.append(self.current_fit)
            if len(self.current_fit) > self.history_size:
                self.current_fit.pop(0)

            if self.best_fit is None:
                self.best_fit = self.current_fit
            else: # Should we use recent_detects to compute the best fit?
                self.best_fit = self.best_fit * self.smooth_factor + self.current_fit * (1. - self.smooth_factor)

            self.detected = True
            self.fails = 0
            self.committed = False
            return True

    def unset(self):
        '''
        Undo last set
        '''
        if not self.committed and len(self.recent_detects) > 0:
            self.committed = True
            self.current = self.recent_detects[-1]
            self.current_fit = self.recent_fits[-1]
            self.recent_detects = self.recent_detects[:-1]
            self.recent_fits = self.recent_fits[:-1]
            return True
        return False

    def commit(self):
        '''
        Commit current set, after this, it cannot be unset
        '''
        self.committed = True

    def curverature(self, poly, y):
        c = abs(2*poly[0])/math.pow((1. + (2*poly[0]*y + poly[1])**2), 3./2.)
        return c

    def x(self, y):
        '''
        Return the x coordinate at y with the current fit
        '''
        assert self.current_fit is not None, "The line has no fit!"
        return np.polyval(self.current_fit, y)

    def bestx(self, y):
        '''
        Return the x coordinate at y with the current fit
        '''
        if self.best_fit is None:
            return np.polyval(self.current_fit, y)
        else:
            return np.polyval(self.best_fit, y)

    def dist(self, another):
        '''
        Compute the minimal and maximal distance with another line
        another: another line
        '''
        if self.current_fit is None: # nothing to compare
            return [1e6, 1e6]
        if isinstance(another, Lane):
            if another.current_fit is None: # nothing to compare
                return [-1, -1]
            another = another.current_fit
        if self.current_fit[0] == another[0]: # should be safe to assume the lines are identical?
            return [0, 0]
        ymin = (another[1]-self.current_fit[1])/(2.*(self.current_fit[0]-another[0]))
        if self.max_y >= ymin and self.min_y <= ymin: # two intercept in lane region
            return [-1, max(abs(self.x(self.max_y)-np.polyval(another, self.max_y)),
                            abs(self.x(self.min_y)-np.polyval(another, self.min_y)))]
        return sorted([abs(self.x(self.max_y)-np.polyval(another, self.max_y)),
                       abs(self.x(self.min_y)-np.polyval(another, self.min_y))])

    def other_side(self, distance, steps=8, bestfit=True):
        '''
        Create the otherside of the lane
        Arguments:
        distance: the distance to the other side
        steps: the number of points the other side will have
        '''
        if self.current is None:
            return None
        points = []
        distance = sorted(distance)
        step = int((self.min_y-self.max_y)/steps)
        distance_step = (distance[0]-distance[1])/steps
        for i in range(steps):
            d = distance_step * i + distance[1]
            y = step * i + self.max_y
            if bestfit and self.best_fit is not None:
                points.append([int(self.bestx(i)) + d, y])
            else:
                points.append([int(self.x(i)) + d, y])
        other = Lane(self.config, self.min_y, self.max_y, self.smooth_factor, self.distance_thresh,
                     self.curverature_threshold)
        other.set(points)
        other.commit()
        return other

class LaneDetector:
    '''
    Lane detector class that use an image processing pipeline to detect lanes
    '''
    def __init__(self, config, camera):
        '''
        camers: the camera used for capturing the images. It must has been calibrated
        config: the configuration
        '''
        # Camera
        self.camera = camera
        # Detection configuration
        self.config = config
        config.crop = sorted(config.crop)
        config.trapezoid_x = sorted(config.trapezoid_x)
        # The ytop and bottom coordinates of the trapezoid
        config.trapezoid_y = sorted(config.trapezoid_y)
        self.trapezoid_top = config.trapezoid_x[2] - config.trapezoid_x[1]
        self.trapezoid_bottom = config.trapezoid_x[3] - config.trapezoid_x[0]
        config.sobel_thresh = sorted(config.sobel_thresh)
        config.magnitude_thresh = sorted(config.magnitude_thresh)
        # Normalized sobel threshold
        self.sobel_thresh = np.float32(config.sobel_thresh)/255.0
        # Normalized sobel mangitude threshold
        self.magnitude_thresh = np.float32(config.magnitude_thresh)/255.0
        # Normalized HLS threshold
        if config.hls_thresh is None:
            self.hls_thresh = None
        else:
            config.hls_thresh = sorted(config.hls_thresh)
            self.hls_thresh = np.float32(config.hls_thresh)/255.0

        if self.config.test:
            print("Detection image height: ", self.config.layer_height*self.config.scan_layers)
            print("Scan threshold", self.config.scan_thresh)

        self.left_lane = None
        self.right_lane = None
        self.reset()
        # Convolution window
        self.convolution = np.ones(self.config.sliding_width)

    def apply_hsv_thresh(self, image):
        '''
        Apply threshold of the white and yellow colors of the HSV colorspace
        image: the image
        Returns: the HSV image, the threshed HSV image, the HSV maskoff image for removing lane change mark
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = None
        maskoff = None
        if self.config.hsv_thresh is not None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for (lower, upper) in self.config.hsv_thresh: # filter range
                mask |= cv2.inRange(hsv, lower, upper)
        if self.config.hsv_maskoff is not None:
            maskoff = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for (lower, upper) in self.config.hsv_maskoff: # filter range
                maskoff |= cv2.inRange(hsv, lower, upper)
        return hsv, mask, maskoff == 0

    def apply_hls_thresh(self, image):
        '''
        Apply the threshold of the saturation of the HLS colorspace
        Returns the HLS threshed image
        '''
        if self.hls_thresh is None:
            return None
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        channel = hls[:, :, 2]
        max_saturation = np.amax(channel)
        hls_thresh = self.hls_thresh * max_saturation
        return (channel >= hls_thresh[0]) & (channel <= hls_thresh[1])

    def apply_threshold(self, image):
        '''
        Apply threshold to the image for extracting lane feature
        image: the image
        Return: lane extracted image, and a tuple of (sobelx, sobely, sobelm, hls, hsv, maskoff) when
        test mode is enabled, or None otherwise
        sobelx: the sobel threshed image for x axis
        sobely: the sobel threshed image for y axis
        sobelm: the sobel magnitude threshed image
        hls: the HLS colorspace threshed image
        hsv: the HSV colorspace threshed image
        maskoff: the HSV colorspace threshed image to mask off dark color due to lane change
        '''
        hsv_image, hsv, maskoff = self.apply_hsv_thresh(image)
        gray = hsv_image[:, :, 2] #cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.config.sobel_kernel))
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.config.sobel_kernel))
        sobelm = np.sqrt(sobelx * sobelx + sobely * sobely)
        sobelx = sobelx/(np.amax(sobelx)+1e-6)
        sobely = sobely/(np.amax(sobely)+1e-6)
        sobelm = sobelm/(np.amax(sobelm)+1e-6)
        extracted = np.zeros_like(gray, dtype=np.uint8)
        hls = self.apply_hls_thresh(image)

        passed = (((sobelx >= self.sobel_thresh[0]) & (sobelx <= self.sobel_thresh[1]) &
                   (sobely >= self.sobel_thresh[0]) & (sobely <= self.sobel_thresh[1])) |
                  ((sobelm >= self.magnitude_thresh[0]) & (sobelm <= self.magnitude_thresh[1])))

        if hls is not None:
            passed = passed | hls
        if hsv is not None:
            passed = passed | (hsv > 0)
        if maskoff is not None:
            passed = passed & maskoff
        extracted[passed] = 1
        if self.config.test:
            hls_out = None
            if hls is not None:
                hls_out = np.zeros_like(hls, dtype=np.uint8)
                hls_out[hls] = 1
            return extracted, (sobelx, sobely, sobelm, hls, hsv, maskoff)
        return extracted, None

    def preprocess(self, image):
        '''
        Proprecess the image for lane detection. It will un-distort the image, then extract lanes features,
        then apply reverse perspective transformation
        images: the image
        Return: undistorted image, transformed image ready for finding lanes, (cropped undistored image,
        lane extracted image, sobelx, sobely, sobelm, hls, hsv, maskoff) when in test mode, or None otherwise
        '''
        image = image[self.config.crop[0]:self.config.crop[1], :, :]
        extracted, extras = self.apply_threshold(image)

        # Mask the extracted line feature alone the existing lane line areas
        mask = None
        if self.left_lane.best_fit is not None:
            mask = utils.line_mask(self.config.image_size, self.left_lane, self.config.crop[0], self.config.crop[1],
                                   40, self.config.scan_width, self.config.scan_width * 1.2)
        if self.right_lane.best_fit is not None:
            utils.line_mask(self.config.image_size if mask is None else mask, self.right_lane, self.config.crop[0],
                                   self.config.crop[1], 40, self.config.scan_width, self.config.scan_width * 1.2)
        if mask is not None:
            extracted = np.bitwise_and(extracted, mask)
        tranformed = self.camera.parallel(extracted)
        if self.config.test:
            return tranformed, (image, extracted) + extras
        else:
            return tranformed, None

    def draw_window(self, image, x, y, level, box_color=[0, 255, 0], color=[255, 0, 0]):
        '''
        Draw a window to show trace of line scan
        Arguments:
        image: the image to draw
        x: x coordinate of the center of the box
        y: y coordinate of the center of the box
        level: the current scan level
        box_coloe: color of the box, default is green
        color: center point color, default is red
        '''
        x = int(x)
        y = int(y)
        half = int(self.config.sliding_width/2)
        ybot = int(image.shape[0] - level * self.config.layer_height)
        ytop = int(ybot - self.config.layer_height)
        image[ytop:ybot, max(0, x-half):min(x+half, image.shape[1]), :] = box_color
        cv2.circle(image, (x, y), 8, color, -1)

    def reset(self):
        '''
        Reset the lines
        '''
        self.left_lane = Lane(self.config, self.config.crop[0], self.config.crop[1], self.config.smooth_factor,
                              self.config.lane_shift_thresh, self.config.curverature_threshold)
        self.right_lane = Lane(self.config, self.config.crop[0], self.config.crop[1], self.config.smooth_factor,
                               self.config.lane_shift_thresh, self.config.curverature_threshold)

    def start_detect(self, image):
        '''
        Find the starting point for lane detection. It incrementally scan the image from the bottom, and increase
        the scan height until the start point is found.
        image: the image
        '''
        # First find the two starting positions for the left and right lane by using np.sum to get the
        # vertical image slice and then np.convolve the vertical image slice with the window template
        # Sum half bottom of image to get slice, could use a different ratio
        half = self.config.sliding_width/2

        # Find the best starting point on the left
        divide = self.config.start_divide
        midium = int(image.shape[1]/2)
        while divide > 1:
            left_hist = np.sum(image[int(image.shape[0]/divide):, :midium], axis=0)
            convolution = np.convolve(self.convolution, left_hist)
            leftx = int(max(np.argmax(convolution) - half, 0))
            divide -= 1
            if convolution[leftx] >= self.config.scan_thresh:
                break

        # Find the best starting point on the right
        divide = self.config.start_divide
        while divide > 1:
            right_hist = np.sum(image[int(image.shape[0]/divide):, midium:], axis=0)
            convolution = np.convolve(self.convolution, right_hist)
            rightx = int(min(np.argmax(convolution)-half+midium, image.shape[1]))
            divide -= 1
            if convolution[rightx - midium] >= self.config.scan_thresh:
                break

        # check the distance between the two points must make sense
        dist = abs(rightx-leftx-(self.config.trapezoid_x[3]-self.config.trapezoid_x[0]))
        if dist > self.config.image_size[1] * 0.05: # this is a bad begining
            if self.config.test:
                print("Bad start at: {0}, {1}, distance: {2}".format(leftx, rightx, dist))
            result = self._previous_start()
            if result is not None:
                return result
        return int(leftx), int(rightx)

    def scan(self, image, previous=None, reset=False):
        '''
        Detect lane lines by scanning the binary image
        Arguments:
        image: the binary image
        previous: x coordinate of the previous starting point (left, right)
        '''
        self.config.layer_height = image.shape[0] / self.config.scan_layers
        left_lane = [] # Store the left lane points
        right_lane = [] # Store the right lane points
        half = int(self.config.sliding_width/2)

        if previous is None:
            leftx, rightx = self.start_detect(image)
        else:
            leftx, rightx = previous[0], previous[1]

        if self.config.test: # Create visualization image
            visual = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        else:
            visual = None
        if self.config.test:
            print("Start center: ", leftx, rightx)
        # Go through each layer looking for max pixel locations
        left_misses = 1
        right_misses = 1
        left_start = None
        right_start = None
        for i in range(0, self.config.scan_layers):
            # convolve the window into the vertical slice of the image
            ybot = int(image.shape[0] - i * self.config.layer_height)
            ytop = int(ybot - self.config.layer_height)
            histogram = np.sum(image[ytop:ybot, :], axis=0)
            # Find the best left centroid by using past left center as a reference
            # Use self.config.sliding_width/2 as offset because convolution signal reference is at
            # right side of window, not center of window. The scan width is increased each time we
            # fail to find a point
            dw = self.config.scan_width - self.config.sliding_width
            left_min = int(max(leftx + half - left_misses*self.config.scan_width, 0))
            left_max = int(min(leftx + half + left_misses*self.config.scan_width, image.shape[1]))
            right_min = int(max(rightx + half - right_misses*self.config.scan_width, 0))
            right_max = int(min(rightx + half + right_misses*self.config.scan_width, image.shape[1]))

            histogram[left_min+dw:left_max-dw] = histogram[left_min+dw:left_max-dw] * 2
            histogram[right_min+dw:right_max-dw] = histogram[right_min+dw:right_max-dw] * 2
            convolution = np.convolve(self.convolution, histogram)

            # argmax will not return valid index if no pixel is on, or there are noise
            # we will set leftx, rightx when the convolution exceed the threshold
            leftx_tmp = np.argmax(convolution[left_min:left_max]) + left_min - half
            rightx_tmp = np.argmax(convolution[right_min:right_max]) + right_min - half

            # TODO: try if using average of on points to compute x and y can yield better fits
            ymid = int(ytop + self.config.layer_height/2)
            if convolution[leftx_tmp + half] >= self.config.scan_thresh:
                # found left lane for that layer, accept the point and set the new leftx
                leftx = leftx_tmp
                left_lane.append([leftx, ymid])
                left_misses = 1
                if self.config.test: # Draw visualization image
                    self.draw_window(visual, leftx, ymid, i)
            elif i == 0: # the first layer, but we could find y, use the one from start
                left_start = [leftx, ymid]
                left_misses = 1
            elif self.config.test:
                if (left_misses + 1) * self.config.scan_width <= self.config.max_scan_width:
                    left_misses += 1
                print("Skip left: ", leftx, ymid, left_min, left_max, convolution[leftx_tmp])

            if convolution[rightx_tmp + half] >= self.config.scan_thresh: # found right lane for that layer
                # found right lane for that layer, accept the point and set the new leftx
                rightx = rightx_tmp
                right_lane.append([rightx, ymid])
                right_misses = 1
                if self.config.test: # Draw visualization image
                    self.draw_window(visual, rightx, ymid, i)
            elif i == 0: # the first layer, but we could find y, use the one from start
                right_start = [rightx, ymid]
                right_misses = 1
            elif self.config.test:
                if (right_misses + 1) * self.config.scan_width <= self.config.max_scan_width:
                    right_misses += 1
                print("Skip right: ", rightx, ymid, right_min, right_max, convolution[rightx_tmp])

            if self.config.test:
                print("Layer center: ", leftx, rightx)

        # TODO: Try the idea of creating one side from the other when one is good and another is bad
        if len(left_lane) == 1 and left_start is not None:
            # only one point, add the start
            left_lane = [left_start] + left_lane
            if self.config.test: # Draw visualization image
                self.draw_window(visual, left_start[0], left_start[1], 0)
        if len(right_lane) ==1 and right_start is not None:
            # only one point, add the start
            right_lane = [right_start] + right_lane
            if self.config.test: # Draw visualization image
                self.draw_window(visual, right_start[0], right_start[1], 0)

        if len(left_lane) >= 2: # we have enough point for a polyline
            left_lane = self.camera.perspective(left_lane)
            left_lane[:, 1] = left_lane[:, 1] + self.config.crop[0]
            if not self.left_lane.set(left_lane):
                if self.config.test:
                    print("Left lane rejected!")
                    print("Left lane: ", self.left_lane.current, left_lane)
            if self.config.test:
                print("Transform left lane: ", left_lane)
                print("Left lane polynomial: ", self.left_lane.current_fit)
        else:
            self.left_lane.set(None)
        if len(right_lane) >= 2: # we have enough point for a polyline
            right_lane = self.camera.perspective(right_lane)
            right_lane[:, 1] = right_lane[:, 1] + self.config.crop[0]
            if not self.right_lane.set(right_lane):
                if self.config.test:
                    print("Right lane rejected!")
                    print("Right lane: ", self.right_lane.current, ", new: ", right_lane)
            if self.config.test:
                print("Transform right lane: ", right_lane)
                print("Right lane polynomial: ", self.right_lane.current_fit)
        else:
            self.right_lane.set(None)

        # Make sure the distance between left and right lane make sense
        dist = self.right_lane.dist(self.left_lane)
        if abs(dist[0]-self.trapezoid_top)/self.trapezoid_top > self.config.width_variation or \
           abs(dist[1]-self.trapezoid_bottom)/self.trapezoid_bottom > self.config.width_variation:
            # The distance does not make sense, undo the set() operation
            self.right_lane.unset()
            self.left_lane.unset()
            # Reset if the number of failure exceeds the threshold
            if not reset and max(self.left_lane.fails, self.right_lane.fails) > self.config.failure_reset_thresh:
                print("Reset detection ...")
                left = self.left_lane
                right = self.right_lane
                self.reset()
                self.scan(image, None, True)
                if self.right_lane.detected and self.left_lane.detected:
                    # Both lanes are detected, verify the distance, it has to be better than the previous
                    dist2 = self.right_lane.dist(self.left_lane)
                    if (dist2[0] - dist[0]) + (dist2[1] - dist[1]) <= 0:
                        return visual
                print("Reset failed, restore previous ... ")
                self.left_lane = left
                self.right_lane = right
        else: # We are good, commit lane set()
            self.right_lane.commit()
            self.left_lane.commit()

        return visual

    def _previous_start(self):
        return None
        if (self.left_lane.current is not None) and (self.right_lane.current is not None):
            if ((self.camera.image_size[0] - self.left_lane.current[0][1]) < self.config.layer_height) and \
               ((self.camera.image_size[0] - self.right_lane.current[0][1]) < self.config.layer_height):
                return self.left_lane.current[0][0], self.right_lane.current[0][0]

    def detect(self, image):
        '''
        Detect lane
        image: the image to detect
        Return: the left and right lanes in format [[Lx0, Lx1, Lx2], [Rx0, Rx1, Rx2]]
        '''
        trasnsformed, extras = self.preprocess(image)
        previous_start = self._previous_start()
        visual = self.scan(trasnsformed, previous_start)
        overlay = utils.draw_lane(image.shape, self.config.crop[0], image.shape[0], 30, self.left_lane,
                                  self.right_lane)
        if self.config.test:
            return overlay, (visual, trasnsformed) + extras
        else:
            return overlay, None
        