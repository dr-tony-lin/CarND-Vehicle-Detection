'''
Implements vehicle detection pipeline
'''
import math
import pickle
import time
import json
from sys import stdout

import numpy as np
import cv2
from joblib import Parallel, delayed
import matplotlib.image as mpimg
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.ndimage.measurements import label

import utils

class VehicleDetector:
    '''
    In order to allow arbritary large samples, and incremental training, the vehicle detector uses SGDClassifier
    to train vehicle detection through SGDClassifier.partial_fit(). The training process divides the samples into
    a number of batches, and partial fit each batch in an epoch. The training process repeats for a given number
    of epoch, and models that pass an accuracy threshold are saved.
    The vehicle_config.config provide configuration parameters to the training and detection process.
    '''
    def __init__(self, config, checkpoint=None):
        self.config = config
        self.scaler = None
        if checkpoint is None:
            self.trained = False
            if config.train.method == 'sgd':
                self.classifier = SGDClassifier(loss=config.sgd.loss, alpha=config.sgd.alpha,
                                                eta0=config.sgd.eta0, epsilon=config.sgd.epsilon,
                                                l1_ratio=config.sgd.l1_ratio,
                                                learning_rate=config.sgd.learning_rate,
                                                penalty=config.sgd.penalty, power_t=config.sgd.power_t,
                                                average=config.sgd.average, fit_intercept=config.sgd.fit_intercept,
                                                shuffle=True, warm_start=False, n_jobs=config.train.cpus)
            elif config.train.method == 'svc':
                self.classifier = SVC(C=config.svc.C, kernel=config.svc.kernel)
            elif config.train.method == 'linearsvc':
                self.classifier = SVC(C=config.svc.C, kernel=config.svc.kernel)
            else:
                raise ValueError("Unsupported classifier: " + config.train.method)
        else:
            self.trained = True
            self.load(checkpoint)

    def save(self, file):
        '''
        Save trained model to file
        file: name of the file to save
        Return: True is model is saved
        '''
        assert self.classifier is not None, "The model has not yet been trained"
        data = {
            "classifier": self.classifier,
            "scaler": self.scaler,
            "colorspace": self.config.colorspace,
            "train_size": self.config.train.size,
            "hog_orientations": self.config.hog.orientations,
            "hog_pixels_per_cell": self.config.hog.pixels_per_cell,
            "hog_cells_per_block": self.config.hog.cells_per_block,
            "hog_channels": self.config.hog.channels,
            "bin_spatial_size": self.config.bin.spatial_size,
            "histogram_bins": self.config.histogram.bins,
            "histogram_range": self.config.histogram.range
        }

        dump = pickle.dumps(data)
        with open(file, 'wb') as fout:
            fout.write(dump)

    def load(self, file):
        '''
        Load trained model from file
        file: name of the file
        '''
        with open(file, 'rb') as fin:
            dump = fin.read()
            data = pickle.loads(dump)
            self.classifier = data["classifier"]
            self.scaler = data["scaler"]
            self.config.colorspace = data["colorspace"]
            self.config.train.size = data["train_size"]
            self.config.hog.orientations = data["hog_orientations"]
            self.config.hog.pixels_per_cell = data["hog_pixels_per_cell"]
            self.config.hog.cells_per_block = data["hog_cells_per_block"]
            self.config.hog.channels = data["hog_channels"]
            self.config.bin.spatial_size = data["bin_spatial_size"]
            self.config.histogram.bins = data["histogram_bins"]
            self.config.histogram.range = data["histogram_range"]
            if isinstance(self.classifier, SGDClassifier):
                self.config.train.method = 'sgd'
            elif isinstance(self.classifier, SVC):
                self.config.train.method = 'svc'

            return self

    def tune(self, samples, labels):
        '''
        Tune the classifier parameters
        samples: the tunning samples
        labels: the tunning labels
        '''
        if not self.trained:
            with self.config as config:
                if config.train.method == 'sgd':
                    parameters = config.sgd.tunning_parameters
                elif config.train.method == 'svc':
                    parameters = config.svc.tunning_parameters
                clf = GridSearchCV(self.classifier, parameters, verbose=1, return_train_score=True)
                self.classifier = clf.fit(samples, labels)
                return clf.best_estimator_, clf.best_params_, clf.best_score_, clf.cv_results_

    def evaluate(self, images, labels):
        '''
        Evaluate the given image samples
        images: the images
        labels: the labels
        '''
        features = self._training_features(images)
        # Normalize features
        if self.scaler is None:
            self.scaler = RobustScaler().fit(features)
        features = self.scaler.transform(features)
        return self.classifier.score(features, labels)

    def train(self, cars, rests, tune=True):
        '''
        Detect vehicles in the image
        image: the image
        Returns: an array of bounding boxes of each detected hehicle
        '''
        car_labels = np.ones(len(cars), dtype=np.uint8)
        rest_labels = np.zeros(len(rests), dtype=np.uint8)
        labels = np.concatenate((car_labels, rest_labels)).T
        samples = np.concatenate((cars, rests))
        trains, tests, train_labels, test_labels = train_test_split(samples, labels,
                                                                    train_size=self.config.train.train_split)
        print("Test splits: ", trains.shape, train_labels.shape, tests.shape, test_labels.shape)

        classes = np.unique(labels)
        # Extract features for test samples
        with self.config as config:
            print("Extracting features for test samples ...")
            test_features = self._training_features(tests)
            if config.test:
                print("Test features: ", test_features.shape)

            # Normalize features
            if self.scaler is None:
                self.scaler = RobustScaler().fit(test_features)
            test_features = self.scaler.transform(test_features)

            if tune or config.train.tune: # Tune training parameters if speficied
                print("Tuning classifier ...")
                self.classifier, best_params, score, _ = self.tune(test_features, test_labels)
                print("Tunned score: ", score, ", parameters: ", json.dumps(best_params))

            print("Start training ...")
            if config.train.method == 'sgd':
                self._sgd_train(trains, train_labels, test_features, test_labels, classes)
            elif config.train.method == 'svc':
                self._svc_train(trains, train_labels, test_features, test_labels, classes)

    def _sgd_train(self, trains, train_labels, test_features, test_labels, classes):
        '''
        Perform SGD training
        trains: the training samples
        train_labels: the training label
        test: the test samples
        test_labels: the test labels
        classes: number of label classes
        tune: true to tune the training parameters first
        '''
        self._incremental_train(trains, train_labels, test_features, test_labels, classes)

    def _svc_train(self, trains, train_labels, test_features, test_labels, classes):
        '''
        Perform SVC training
        trains: the training samples
        train_labels: the training label
        test: the test samples
        test_labels: the test labels
        classes: number of label classes
        tune: true to tune the training parameters first
        '''
        self._train_all(trains, train_labels, test_features, test_labels)

    def _incremental_train(self, trains, train_labels, test_features, test_labels, classes):
        '''
        Perform incremental training:
        1. The training samples are divided into batches
        2. The training go through a specified number of epochs
        3. In each epoch, the training samples are shuffled first, then the samples are broken into batch
        4. Each batch is feed into the classifier
        5. Once all batches are processed, the classifier is then tested with the test samples for accuracy
        6. A checkpoint is saved when the test accuracy is above a threshold
        trains: the training samples
        train_labels: the training label
        test: the test samples
        test_labels: the test labels
        classes: number of label classes
        tune: true to tune the training parameters first
        '''
        with self.config as config:
            start_time = time.time()
            # Go through each epoch
            for epoch in range(config.train.epochs):
                # Shuffle samples as the classifier will not shuffle in partial fit
                trains, train_labels = shuffle(trains, train_labels)
                epoch_start = time.time()
                # Loop through the batches
                for index in range(0, len(trains), config.train.batch_size):
                    end = min(index + config.train.batch_size, len(trains))
                    batch_samples = trains[index:end]
                    batch_labels = train_labels[index:end]
                    # Extract batch samples' features
                    features = self._training_features(batch_samples)

                    # Normalize features
                    if self.scaler is None:
                        self.scaler = RobustScaler().fit(features)
                    features = self.scaler.transform(features)
                    # Perform partial fit
                    self.classifier.partial_fit(features, batch_labels, classes)
                    if config.test:
                        stdout.write("Epoch progress: %d%%   \r" % (100 * end / len(trains)))
                        stdout.flush()

                epoch_time = int(time.time() - epoch_start)
                accuracy = self.classifier.score(test_features, test_labels)
                print("Epoch {0} accuracy: {1}, time {2} seconds".format(epoch, accuracy, epoch_time))
                if accuracy >= 0.96:
                    print("Saving model ...")
                    self.save(config.train.checkpoint + '-{0}-{1}-{2}-{3}.mdl'.format(config.train.method,
                                                                                      config.colorspace, epoch,
                                                                                      int(accuracy*1000)))
            print("Training time for {0} epochs: {1} seconds".format(config.train.epochs,
                                                                     int(time.time() - start_time)))

    def _train_all(self, trains, train_labels, test_features, test_labels):
        '''
        Perform a single training with entire sample:
        trains: the training samples
        train_labels: the training label
        test: the test samples
        test_labels: the test labels
        classes: number of label classes
        tune: true to tune the training parameters first
        '''
        with self.config as config:
            start_time = time.time()
            trains, train_labels = shuffle(trains, train_labels)
            # Extract batch samples' features
            features = self._training_features(trains)

            # Normalize features
            if self.scaler is None:
                self.scaler = RobustScaler().fit(features)
            features = self.scaler.transform(features)
            # Perform partial fit
            self.classifier.fit(features, train_labels)
            end_time = time.time()
            accuracy = self.classifier.score(test_features, test_labels)
            self.save(config.train.checkpoint + '-{0}-{1}-{2}.mdl'.format(config.train.method, config.colorspace,
                                                                          int(accuracy*1000)))
            print("Training time: {0} seconds, accuracy: {1:.3}".format(int(end_time - start_time), accuracy))

    def _process_heatmap(self, draw, heatmap):
        # Threshold the heatmap
        if self.config.current_image is not None:
            mpimg.imsave(self.config.test_output + "heatmap-{0}".format(self.config.current_image), heatmap)
        heatmap[heatmap <= self.config.heatmap_threshold] = 0
        # Label the heatmap
        labels = label(heatmap)
        if labels[1] > 0:
            # Draw bounding box around the labels
            if self.config.test:
                bboxes = []
            else:
                bboxes = None
            for car_no in range(1, labels[1]+1):
                # Find pixels with each car_number label value
                nonzero = (labels[0] == car_no).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                xmin = np.min(nonzerox)
                ymin = np.min(nonzeroy)
                xmax = np.max(nonzerox)
                ymax = np.max(nonzeroy)
                # Define a bounding box based on min/max x and y
                bbox = ((xmin, ymin), (xmax, ymax))

                if self.config.test:
                    print("Label min, max: ", xmin, ymin, xmax, ymax)
                    bboxes.append(bbox)

                # Draw the box on the image
                cv2.rectangle(draw, bbox[0], bbox[1], self.config.border_color, self.config.border_thickness)

            if self.config.test:
                if labels[1] > 0:
                    print("Bounding boxes (", labels[1], "): ", bboxes)

    def _predict(self, features, cpu):
        '''
        Parallel prediction task
        features: the features to normalize and perdict
        cpu: index of the cpu
        '''
        # Compute the split boundary
        split_size = int(math.ceil(features.shape[0] / self.config.predict_cpus))
        split_start = cpu * split_size
        split_end = min(split_start + split_size, features.shape[0])
        if split_start >= split_end: # No sample to process
            return []
        # Get the sample split
        split_features = features[split_start:split_end]
        # Normalize the features
        split_features = self.scaler.transform(split_features)
        # Perform perdiction
        predicts = self.classifier.predict(split_features)
        # Offset the matched location indices
        predicts = predicts.nonzero()[0] + split_start
        if self.config.test:
            print("Batch split: ", split_start, split_end, ", matches: ", predicts)
        return predicts

    def detect(self, image):
        '''
        Detect vehicles in the image
        image: the image
        Returns: the overlay image
        '''
        with self.config as config, Parallel(n_jobs=config.predict_cpus, backend="threading") as par:
            if config.test:
                print("Start vehicle detection: ", image.shape)

            start = time.time()

            # apply color conversion and normalization
            image = utils.convert_color(image, config.colorspace)

            # Initialize the overlay image, and heatmap
            overlay = np.zeros(image.shape, dtype=np.uint8)
            heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float)

            # Loop all sliding configuration, for each configuration:
            # 1. Compute the scaling factor between the training image size and the sliding window size
            # 2. Crop the image to the detection range
            # 3. Scale the image with the scaling factor
            # 4. Compute all sliding windows and extract the detection features for all sliding window,
            #    each windows has a row of features
            # Split the samples among the designated threads, and run the followings in parallel on the splits
            #     5. Normalize the features
            #     6. Perform perdiction
            # 7. Compute the heatmap
            # 8. Compute the label of the heatmap
            # 9. Draw bounding box on the overlap image
            for slide in config.slidings:
                # Compute the scaling factor between the training image size and the sliding window size
                # Crop the image to the detection range
                cropped = image[slide[1][0]:slide[1][1], :, :]

                # Compute the scaling factor
                if config.train.size[1] != slide[0][1] or config.train.size[0] != slide[0][0]:
                    sx = slide[0][0] / float(config.train.size[0])
                    sy = slide[0][1] / float(config.train.size[1])
                    # Scale the image with the scaling factor
                    size = (int(cropped.shape[1]/sx), int(cropped.shape[0]/sy))
                    resized = cv2.resize(cropped, size)
                else: # No scaling is needed
                    size = config.train.size[0]
                    resized = cropped

                if config.test:
                    print("Slide:", slide, " scale: ", (sx, sy), ", size: ", size, " cropped resized shape: ",
                          resized.shape)

                # Extract the detection features for all sliding window, each windows has a row of features
                features, windows, image_hog = self._detection_features(resized)
                features = np.array(features)
                windows = np.array(windows)

                # Compute windows geometry in the original image space

                windows[:, :, 0] = np.int32(windows[:, :, 0] * sx)
                windows[:, :, 1] = np.int32(windows[:, :, 1] * sy + slide[1][0])

                if self.scaler is None:
                    self.scaler = RobustScaler().fit(features)

                ones = np.concatenate(par(delayed(self._predict)(features, cpu) for cpu in range(config.predict_cpus)))

                if config.test:
                    matched_windows = []

                # Compute the heatmap
                for idx in ones:
                    win = windows[int(idx)]
                    heatmap[win[0][1]:win[1][1], win[0][0]:win[1][0]] += 1
                    if config.test:
                        matched_windows.append(win)

                if config.test:
                    print("Detection time: {0}".format(time.time() - start))
                    print("Features: ", features.shape)
                    print("Predictions matches (", len(ones), "): ", ones)
                    print("Windows: ", len(windows), ", matched: ", matched_windows)
                    if image_hog is not None:
                        mpimg.imsave(config.test_output + "hog{0}-{1}".format(slide[0][0], config.current_image),
                                     image_hog)
            self._process_heatmap(overlay, heatmap)
            return overlay

    def _training_features(self, images):
        '''
        Extract features of the given images
        Arguments:
        images: the images, can be an array of image files or RGB images
        '''
        result = []
        with self.config as config, self.config.hog as hog, self.config.histogram as hist:
            for image in images: # Iterate through the list of images
                if isinstance(image, str):
                    image = mpimg.imread(image)
                # apply color conversion if other than 'RGB'
                image = utils.convert_color(image, config.colorspace)
                # Get hog features
                features, _ = self._extract_hog(image, train=True, visualise=False)

                # Get color features if bin_spatial_size is specified
                if config.bin.spatial_size is not None:
                    features = np.concatenate((features, utils.bin_spatial(image, size=config.bin.spatial_size)))

                # Get histogram features if histogram_bins is specified
                if config.histogram.bins is not None:
                    features = np.concatenate((features, utils.color_histogram(image, bins=hist.bins,
                                                                               range=hist.range)))
                result.append(features)
            result = np.vstack(result)
        return result

    def _extract_hog(self, image, train=False, visualise=False):
        '''
        Extract hog features
        image: the image
        train: True for training, False for detection
        '''
        hog_channels = []
        with self.config as config, self.config.hog as hog:
            if config.hog.channels == 'ALL':
                hog_image = None
                for channel in range(image.shape[2]):
                    features, him = utils.hog(image[:, :, channel],
                                              orientations=hog.orientations,
                                              pixels_per_cell=hog.pixels_per_cell,
                                              cells_per_block=hog.cells_per_block,
                                              visualise=visualise, feature_vector=train)
                    if train:
                        hog_channels = np.concatenate((hog_channels, features))
                    else:
                        hog_channels.append(features)
                    if visualise:
                        if hog_image is None:
                            hog_image = him
                        else:
                            hog_image += him
            elif len(image.shape) == 2: # Grayscale image
                features, hog_image = utils.hog(image,
                                                orientations=hog.orientations,
                                                pixels_per_cell=hog.pixels_per_cell,
                                                cells_per_block=hog.cells_per_block,
                                                visualise=visualise, feature_vector=train)
                if train:
                    hog_channels = features
                else:
                    hog_channels.append(features)
            else:
                features, hog_image = utils.hog(image[:, :, hog.channels],
                                                orientations=hog.orientations,
                                                pixels_per_cell=hog.pixels_per_cell,
                                                cells_per_block=hog.cells_per_block,
                                                visualise=visualise, feature_vector=train)
                if train:
                    hog_channels = features
                else:
                    hog_channels.append(features)
        return hog_channels, hog_image

    def _detection_features(self, image):
        '''
        Extract detection features of the given image
        Arguments:
        image: the image, it should has been converted to the target colorspace
        windows: the windows to extract the features
        orientations: hog orientation to extract
        pixels_per_cell: number of pixel per hog cell
        cells_per_block: number of cells per hog block
        hog_channel: the channel to extract for hog
        bin_spatial_size: the number of color bins to extract
        histogram_bins: the number of histogram bins
        histogram_range: the histogram value range
        '''
        result = []
        # Get hog features
        hog_channels = []
        with self.config as config, self.config.hog as hog, self.config.histogram as hist:
            # Extract hog features for the entire image
            hog_channels, hog_image = self._extract_hog(image, train=False, visualise=True)

            blocks = ((image.shape[1] // hog.pixels_per_cell) - hog.cells_per_block + 1,
                      (image.shape[0] // hog.pixels_per_cell) - hog.cells_per_block + 1)

            blocks_per_window = ((config.train.size[0]//hog.pixels_per_cell)-hog.cells_per_block+1,
                                 (config.train.size[1]//hog.pixels_per_cell)-hog.cells_per_block+1)
            cells_per_step = (config.slide_step[0]//hog.pixels_per_cell,
                              config.slide_step[1]//hog.pixels_per_cell)
            steps = ((blocks[0] - blocks_per_window[0]) // cells_per_step[0] + 1,
                     (blocks[1] - blocks_per_window[1]) // cells_per_step[1] + 1)
            if config.test:
                print("Hog sliding: ", blocks, blocks_per_window, cells_per_step, steps)
            windows = []
            result = []
            show = []
            title = []
            for yb in range(steps[1]):
                yc = yb * cells_per_step[1]
                for xb in range(steps[0]):
                    xc = xb * cells_per_step[0]
                    # Extract HOG for this patch
                    hog_features = []
                    for channel in hog_channels:
                        hog_features.append(channel[yc:yc+blocks_per_window[1], xc:xc+blocks_per_window[0]].ravel())
                    hog_features = np.hstack(hog_features)

                    xleft = xc * hog.pixels_per_cell
                    ytop = yc * hog.pixels_per_cell
                    if config.test:
                        print("Extract windows: ", (xc, yc), (xc+blocks_per_window[0], yc+blocks_per_window[1]),
                              (xleft, ytop), (xleft+config.train.size[0], ytop+config.train.size[1]))
                    windows.append(((xleft, ytop), (xleft+config.train.size[0], ytop+config.train.size[1])))

                    # Get the sub image
                    subimg = image[ytop:ytop+config.train.size[1], xleft:xleft+config.train.size[0]]
                    # Extract bin spatial
                    bin_features = utils.bin_spatial(subimg, size=config.bin.spatial_size)
                    #extract histogram
                    histogram_features = utils.color_histogram(subimg, bins=hist.bins, range=hist.range)
                    # Put all features together in an array
                    features = np.hstack((hog_features, bin_features, histogram_features)).reshape(1, -1)
                    # Add the feature to the result set
                    result.append(features)

                    if config.test and config.test_visualize: # For visualization
                        show.append(utils.to_rgb(np.uint8(subimg*255), config.colorspace))
                        show.append(hog_image[ytop:ytop+config.train.size[1], xleft:xleft+config.train.size[0]])
                        title.append("({0},{1}), ({2},{3})".format(xleft, ytop, xleft+config.train.size[0],
                                                                   ytop+config.train.size[1]))
                        title.append("HOG")
                        if len(show) >= 36:
                            utils.show(images=show, dim=(4, 9), titles=title)
                            show = []
                            title = []
        result = np.vstack(result)
        return result, windows, hog_image
