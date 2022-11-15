#!/usr/bin/python3
import jetson_inference
import jetson_utils
import argparse


# set up two command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

# load the image specified on the command line
img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)

# classify the image
class_idx, confidence = net.Classify(img)

# get the class description out and print it
class_desc = net.GetClassDesc(class_idx)
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))



