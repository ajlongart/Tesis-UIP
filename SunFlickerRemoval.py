import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

# built-in module
import sys

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", frame)
    
if __name__ == '__main__':
	#Constuccion del parse y del argumento
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--videoOri", required = True, help = "Video de Entrada")
	args = vars(ap.parse_args())

	def nothing(*arg):
		pass

	# Capture video
	video = cv2.VideoCapture(args["videoOri"])

	while(video.isOpened()):
	    ret, frame = video.read()
	
	    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	    cv2.imshow('frame',frame)

	    frameCopy = frame.copy()
	    hsv = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2HSV)

	    cv2.namedWindow('framehsv',cv2.WINDOW_NORMAL)
	    cv2.imshow('framehsv',hsv)

	    copyCrop = frame.copy()
	    #imgCrop = copyCrop[100:800,100:500]

	    cv2.setMouseCallback("frame", click_and_crop)

	    if len(refPt) == 2:
	    	roi = copyCrop[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	    	cv2.imshow("ROI", roi)

	
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	
	video.release()
	cv2.destroyAllWindows()
