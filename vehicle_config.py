'''
Configuration for Vehicle Detection
'''
import os

class Subscriptable(type):
    '''
    Meta class for allowing subscription
    '''
    def __getitem__(cls, x):
        return getattr(cls, x)

class Config:
    '''
    Train parameters
    '''
    __metaclass__ = Subscriptable
    def __exit__(self, *args):
        pass

    def __enter__(self, *args):
        return self

class VehicleConfig(Config):
    '''
    Config class
    '''
    def __init__(self):
        self.train = Config()
        self.svc = Config()
        self.sgd = Config()
        self.cnn = Config()
        self.hog = Config()
        self.bin = Config()
        self.histogram = Config()
        self.predict_cpus = 4
        # Threshold of heatmap
        self.heatmap_threshold = 3
        # Color of car's bounding boxes
        self.border_color = [0, 0, 255]
        #thickness of color's bounding boxes
        self.border_thickness = 6
        # Slidings, each sliding is given as ((width, height), (top, botton))
        self.slidings = [#((378, 378), (302, 680)),
            ((256, 256), (360, 680)),
            ((128, 128), (320, 672)),
            ((64, 64), (386, 514))]#,
            #  ((32, 32), (422, 486))]
        self.slide_step = (16, 16)

config = VehicleConfig()

# True to produce more messages for testing purpose
config.test = False
config.test_visualize = False
config.test_output = "test_outputs/"

# The colorspace to use for vehicle detection, can be: HSV, HLS, YCrCb, YUV, and LUV
config.colorspace = 'YCrCb'

# Hog orientations
config.hog.orientations = 12
config.hog.pixels_per_cell = 8
config.hog.cells_per_block = 2
config.hog.channels = 'ALL'
config.bin.spatial_size = (32, 32)
config.histogram.bins = 64
config.histogram.range = [0, 255]

# Training parameters
# Training method, can be 'sgd', 'svc'
config.train.method = "sgd"
# The vehicle images' file filter
config.train.vehicles = "./vehicles/**/*.png"
# The non vehicle images' file filter
config.train.non_vehicle = "./non-vehicles/**/*.png"
# True to tune parameters first
config.train.tune = False
# Size of images used in training, (width, height)
config.train.size = (64, 64)
# Number of cpus to use for the process
config.train.cpus=4
# Checkpoint base name
config.train.checkpoint="model"
# The batch size
config.train.batch_size = 2500
# The portion of samples to be used for training
config.train.train_split = 0.85
# The number of epochs to train the detector
config.train.epochs = 50
# SGD loss, 'hinge' for SVM
config.sgd.loss = 'hinge'
# SGD alpha, result of the parameter tuning
config.sgd.alpha = 0.0002
# SGD eta0, not used by 'optimal' schedule
config.sgd.eta0 = 0.0
# SGD epsilon, result of the parameter tuning
config.sgd.epsilon = 0.4
# SGD l1_ratio, 0 is equilevant to l2 penalty
config.sgd.l1_ratio = 0
# SGD learning rate
config.sgd.learning_rate = 'optimal'
# SGD loss penalty
config.sgd.penalty = 'l2'
# SGD learning rate decay exponent
config.sgd.power_t = 0.75
# SGD disable average on SGD weights
config.sgd.average = False
# SGD fit_intercept disable as data has been centered
config.sgd.fit_intercept = False
# SGD tuning parameters
config.sgd.tunning_parameters = {
    'alpha': [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001],
    'epsilon': [0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5],
    'power_t': [ 0.4, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
}

config.svc.kernel = 'rbf'
config.svc.C = 3
config.svc.tunning_parameters = {
    'kernel': ('linear', 'rbf'),
    'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

if not os.path.exists(config.test_output):
    os.makedirs(config.test_output)
