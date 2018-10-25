import cv2
import numpy as np
from model import *

def predict(image,im_th):


	# cv2.imshow("sdfasfs",im_th)
	model = load_model()

	npa,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	points = []
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# im_gray = cv2.GaussianBlur(img, (5,5), 1)
	# im_th = cv2.adaptiveThreshold(img, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY_INV , 5,2)
	cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV,im_th)
	cv2.imshow('th_',im_th)
	# im_th = img
	# print(rects)

	for (x,y,w,h) in rects:
		if w*h > 100 and w<h*3 or h > w*5 :
			if h > w*2:
				x -= int(h/2)
				y -= 2
				w = h
				h += 2
			points.append((x,y,w,h))
	
	
	dtype = [('x',int),('y',int),('w',int),('h',int)]
	points = np.array(points, dtype=dtype)
	sortedpoints = np.sort(points,order='x')
	# sortedpoints = np.sort(sortedpoints,order='y')
	result = ''

	for (x,y,w,h) in sortedpoints:
		try:
			temp = cv2.resize(im_th[y-20:y+h+20,x-20:x+w+20],(28,28))
		except Exception:
			continue
		
		temp = np.array(temp).reshape(1,1,28,28)
		temp = temp/255
		p= model.predict(temp)[0]
		
		r = np.where(p == p.max())[0][0]
		if r == 10:
			print('ignore')
		else:
			result += str(r) 
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0),2) 
			cv2.putText(image, str(r), (x,y-10),cv2.FONT_HERSHEY_PLAIN , 2, (0,255,0),2)
	cv2.imshow('img',image)
	print(result)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
