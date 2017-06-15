'''
Lane detetion
'''
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from camera import Camera
from vehicle_config import config as vehicle_config
from lane_config import config as lane_config
import utils
from lane_detection import LaneDetector
from vehicle_detection import VehicleDetector

clip_name = None
clip_seq = 0
total_frames = 0
camera = Camera(lane_config)
lane_detector = LaneDetector(lane_config, camera)
vehicle_detector = None
executor = ThreadPoolExecutor(2)
lane_config.test = False
vehicle_config.test = False

def process_image(image):
    '''
    Callback from video clip
    '''
    global clip_name, clip_seq, detector, total_frames

    # Save the image if instructed to do so
    if clip_name:
        mpimg.imsave(test_videos_output + "{0}{1}.jpg".format(clip_name, clip_seq), image)
    # Undistort the image
    image = camera.undistort(image)
    # Schedule lane detection on a thread
    land_future = executor.submit(lambda img: lane_detector.detect(img), image)
    # Schedule vehicle detection on a thread
    vehicle_future = executor.submit(lambda img: vehicle_detector.detect(img), image)
    # Wait for the detections to complete
    while land_future.running() and vehicle_future.running():
        time.sleep(1) # sleep 1 seconds
    # Draw the detection result on the image
    image = utils.weighted_img(image, land_future.result()[0])
    image = utils.weighted_img(image, vehicle_future.result(), α=0.3, β=0.7)
    if clip_name:
        mpimg.imsave(test_videos_output + "{0}{1}-detect.jpg".format(clip_name, clip_seq), image)
        clip_seq += 1
    total_frames += 1
    return image

if __name__ == '__main__':
    '''
    Process command line arguments, and kick off the video processing pipeline
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest="model", default="xmodel.mdl")
    parser.add_argument('-t', '--threads', dest="threads", default=4)
    parser.add_argument('-o', '--output', dest="output", default="outputs")
    parser.add_argument('-v', '--verbose', dest="verbose", default=False)
    parser.add_argument('-i', '--input', dest="mp4", default="*.mp4")
    args = parser.parse_args()

    vehicle_detector = VehicleDetector(vehicle_config, args.model)
    lane_config.test = args.verbose
    vehicle_detector.predict_cpus = args.threads
    test_videos_output = args.output + "/"

    total_time = 0
    # Loop through all mp4 images
    for name in glob.glob(args.mp4):
        print("Processing: {} ...".format(name))
        # Reset lane detector first
        lane_config.set(name)
        lane_detector.reset()
        # The output mp4 file
        output = test_videos_output + name
        # Set clip info if video clips should be saved
        if lane_config.save_video_images:
            clip_name = name
            clip_seq = 0
        # Open the vider clip
        clip = VideoFileClip(name)
        start = time.time()
        # Process the clip
        new_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
        new_clip.write_videofile(output, audio=False)
        total_time += time.time() - start

    # Shutdown the executor
    executor.shutdown()
    print("Processed {0} frames in {1:0.2} seconds, rate: {2} frame/sec".format(total_frames, total_time,
                                                                                total_frames/total_time))
