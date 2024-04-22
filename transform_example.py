#!/usr/bin/python3
from transform import four_point_transform
import numpy as np
import cv2
import argparse

# construct the argument parse and parse args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image file")
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
args = vars(ap.parse_args())

# load the image and grab the source coordinates
# from the command line
# NOTE: using the 'eval' function is bad form, but for this
# example, it's the easiest way to parse comma separated
# coordinates
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

# apply the four point transform to obtain a "birds eye view"
# of the image
warped = four_point_transform(image, pts)

# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)

# $ python transform_example.py --image images/example_02.png --coords "[(101, 185), (393, 151), (479, 323), (187, 441)]"