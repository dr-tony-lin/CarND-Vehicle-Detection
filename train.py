'''
Train vehicle detector
'''
import glob

from vehicle_config import config as vehicle_config
from vehicle_detection import VehicleDetector

cars = glob.glob(vehicle_config.train.vehicles, recursive=True)
others = glob.glob(vehicle_config.train.non_vehicle, recursive=True)
vehicle_config.test = False
print("Number of car samples {0}, others: {1}".format(len(cars), len(others)))

detector = VehicleDetector(vehicle_config)
detector.train(cars, others, vehicle_config.train.tune)
detector.save("model.mdl")
