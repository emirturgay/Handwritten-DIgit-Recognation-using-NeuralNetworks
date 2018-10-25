import cv2
import numpy as np

def morphology(img):
		
	thimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thimg = cv2.Canny(img, 140, 250)
	# cv2.imshow("sdf",thimg)
	# erode = cv2.erode(thimg,np.ones((3,3),np.uint8))
	# dilute   = cv2.dilate(erode, np.ones((3,3),np.uint8))
	# cv2.imshow('erode', erode)
	# cv2.imshow('d', dilute)

	closing = cv2.morphologyEx(thimg, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
	# cv2.imshow("after_closing",closing)
	dilute   = cv2.dilate(closing, np.ones((3,3),np.uint8))
	dilute = cv2.morphologyEx(dilute, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

	# cv2.imshow("after_dilute",dilute)
	return dilute










