'''
Provides Camera class for distoration correction and perspective transformation
'''
import glob
import math
import numpy as np
import cv2
import matplotlib.image as mpimg

class Camera:
    '''
    Provide camera calibration and transformation
    '''
    def __init__(self, config):
        # mtx of the camera
        self.mtx = None
        # dist of the camera
        self.dist = None
        # Perspective transformation matrix to birds eye
        self.trans = None
        # Inverse perspective transformation matrix to the projection space
        self.invtrans = None
        # Height of the birds eye image to transform to
        self.dest_height = 0
        self.config = config

        if config.crop is not None:
            self.image_size = (config.crop[1] - config.crop[0], config.image_size[1])
            if config.trapezoid_y is not None:
                self.trapezoid_y = [config.trapezoid_y[0] - config.crop[0], config.trapezoid_y[1] - config.crop[0]]
            else:
                self.trapezoid_y = [config.trapezoid_y[0] - config.crop[1], config.trapezoid_y[1] - config.crop[1]]
        else:
            self.image_size = config.image_size
            self.trapezoid_y = config.trapezoid_y

        print("Calibrating camera ...")
        ret, mtx, dist = self.calibrate(config.calibration_folder + "*.jpg",
                                        chessboard_shape=config.chessboard_shape)

        assert ret, "Failed to calibrate camera!"
        if config.test:
            print("Calibrated undistortion params: ", mtx, dist)

        imgpoints, objpoints = self.set_transformation(self.config.layer_height*self.config.scan_layers,
                                                       config.trapezoid_x, self.trapezoid_y)
        if config.test:
            print("Image points:")
            print(imgpoints)
            print("Object points:")
            print(objpoints)
            print("Transformation matrix:")
            print(self.trans)
            print("Inverse transformation matrix:")
            print(self.invtrans)

    def set_transformation(self, dest_height=200, x=[275, 565, 720, 1045], y=[470, 680]):
        '''
        Set perspective transformation
        Arguments:
        dest_height: height of the destination image
        x: the x coordinates of the source trapezoid that will be transformed to a rectangle by the transformation
        y: the top and base y coordinate of the source trapezoid
        '''
        self.dest_height = dest_height
        imgpoints = np.float32([[x[1], y[0]], [x[2], y[0]], [x[3], y[1]], [x[0], y[1]]])
        objpoints = np.float32([[imgpoints[3][0], 0], [imgpoints[2][0], 0],
                                [imgpoints[2][0], dest_height], [imgpoints[3][0], dest_height]])
        self.trans = cv2.getPerspectiveTransform(imgpoints, objpoints)
        self.invtrans = cv2.getPerspectiveTransform(objpoints, imgpoints)
        return imgpoints, objpoints

    def calibrate(self, images, chessboard_shape):
        '''
        Calibrate camera
        Arguments:
        images: chessboard calibration images folder and file name pattern, e.g. camera_cal/*.jpg
        chessboard_shape: shape (number grids) of the chessboard
        '''
        objpoints = []
        imgpoints = []
        objp = np.zeros((chessboard_shape[0]*chessboard_shape[1], 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)
        for name in glob.glob(images):
            image = cv2.cvtColor(mpimg.imread(name), cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(image, chessboard_shape, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image.shape[::-1], None, None)
        if ret:
            self.mtx = mtx
            self.dist = dist
        return ret, mtx, dist

    def undistort(self, image):
        '''
        Un-distort the image
        image: the image to un-distort
        '''
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def parallel(self, image):
        '''
        Perform parallel transformion on the image
        image: the image to transform
        '''
        image = cv2.warpPerspective(image, self.trans, (image.shape[1], self.dest_height), flags=cv2.INTER_LINEAR)
        return image

    def perspective(self, points):
        '''
        Perform perspective transform for the points
        points: the points to transform
        '''
        transformed = cv2.perspectiveTransform(np.float32([points]), self.invtrans)
        return transformed[0]

    def preprocess(self, image):
        '''
        Proprocess the imageL: first undistort the image, then crop the image
        image: the image
        Return: the proprocessed image
        '''
        undistorted = self.undistort(image)
        image = undistorted[self.config.crop[0]:self.config.crop[1], :, :]
        return image
