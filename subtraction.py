from __future__ import print_function
import cv2 as cv
import argparse

def backgroundSubtraction(input):
    backSub = cv.createBackgroundSubtractorMOG2()
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(input))
    if not capture.isOpened():
        print('Unable to open: ' + input)
        exit(0)

    iter=0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        # fgMask = backSub.apply(frame)
        return backSub.apply(frame)
    
# backgroundSubtraction(args.input)