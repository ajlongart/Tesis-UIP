import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

import scipy
import scipy.fftpack
import pylab

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

def applyFFT(frames): #https://github.com/rohanraja/respivision/blob/master/fft.py
	fps = 100	#SampleLength https://github.com/rohanraja/respivision/blob/master/parameters.py
	n = roi.shape[0]
	t = np.linspace(0,float(n)/fps, n)

	signal = abs(scipy.fft(roi[0]))
	freqs = scipy.fftpack.fftfreq(roi[0].size, t[1]-t[0])
	freqs = scipy.fftpack.fftshift(freqs)
	signals = scipy.fftpack.fftshift(signal)

	return signals, freqs
    
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
	
	    #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	    #cv2.imshow('frame',frame)

	    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	    cv2.imshow('frame',frame_gray)

	    copyCrop = frame_gray.copy()

	    cv2.setMouseCallback("frame", click_and_crop)

	    if len(refPt) == 2:
	    	roi = copyCrop[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	    	cv2.imshow("ROI", roi)

	    	fft_roi, freq = applyFFT(roi)
		
	    	pylab.plot(freq,20*np.log10(fft_roi))
	    	pylab.grid()
	    	pylab.show()

	    	continue
	
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	
	video.release()
	cv2.destroyAllWindows()
