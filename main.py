import numpy as np
import pandas as pd
import pickle,sys
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential,model_from_yaml
from keras.layers import Dense,Dropout,Flatten
from keras.utils  import np_utils
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

np.random.seed(7)
num_classes=0

def load_data():
	df = pd.read_csv('./training-data/train.csv')
	y = df['label']
	x = df.drop('label',1)
	
	del df
	
	y = np.array(y)
	x = np.array(x)
	
	train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2)
	del x,y
	
	print('complete')
	train_x = train_x.reshape(len(train_x),1,28,28).astype('float32')
	test_x  = test_x.reshape(len(test_x),1,28,28).astype('float32')

	train_x = train_x/255
	test_x  = test_x/255

	train_y = np_utils.to_categorical(train_y)
	test_y = np_utils.to_categorical(test_y)
	
	num_classes = test_y.shape[1]
	return train_x,test_x,train_y,test_y

def createmodel():
	model = Sequential()
	model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation='relu' ))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128,activation='relu'))
	model.add(Dense(10,activation='softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	return model

def train():
	print('Loading data...',end='')
	
	train_x,test_x,train_y,test_y = load_data() #data loading
	
	print('Done')
	
	model = createmodel()	#model loading
	
	print('Training...\n\n')

	model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=20,batch_size=200,verbose=2) # fitting or training model on data
	
	print('\n\nTraning complete')
	print('testing on test data..',end='')
	
	scores = model.evaluate(test_x,test_y,verbose=0) # % score of model on test set
	
	print('Done')
	print('CNN Error:'+str(100-scores[1]*100))

	############## model saving#################### 
	model_yaml = model.to_yaml()
	with open('model.yaml','w') as yaml_file:
		yaml_file.write(model_yaml)

	model.save_weights('model.h5')
	print('model saved')
	#################################################

def load_model():
	with open('model.yaml','r') as f:
		model_file = f.read()
	model = model_from_yaml(model_file)
	model.load_weights('model.h5')
	return model


def predict(imgpath,mode = 0):
	'''mode = 0 predict mode 
	   mode = 1 debug   mode
	'''
	if imgpath == None or imgpath == '':
		imgpath = './test/photo_1.jpg'
	model = load_model()
	image = cv2.imread(imgpath)

	img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	if img is None:
		print('photo does not exist')
		return
	im_gray = cv2.GaussianBlur(img, (5,5), 1)
	im_th = cv2.adaptiveThreshold(im_gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY_INV , 5,2)
	
	if mode == 1:
		cv2.imshow('wame', im_th)

	
	npa,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	points = []
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	# print(rects)

	for (x,y,w,h) in rects:
		if w*h > 100 and w<h*3 or h > w*5 :
			if h > w*2:
				x -= int(h/2)
				y -= 2
				w = h
				h += 2
			if mode ==1:
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3) 
			points.append((x,y,w,h))
	
	if mode == 1:		
		cv2.imshow('winname', image)
	
	dtype = [('x',int),('y',int),('w',int),('h',int)]
	points = np.array(points, dtype=dtype)
	sortedpoints = np.sort(points,order='x')
	# sortedpoints = np.sort(sortedpoints,order='y')
	result = ''

	for (x,y,w,h) in sortedpoints:
		try:
			temp = cv2.resize(im_th[y-10:y+h+10,x-10:x+w+10],(28,28))
		except Exception:
			continue
		if mode ==1:
			cv2.imshow('temp', temp)
		temp = np.array(temp).reshape(1,1,28,28)
		temp = temp/255
		p= model.predict(temp)[0]
		if p.max() < 0.1:
			print('ignore')
		else:
			r = np.where(p == p.max())[0][0]
			# print(p)
			result += str(r) 
			if mode == 1:
				print(r)
				cv2.waitKey(0)
		
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0),2) 
			cv2.putText(image, str(r), (x,y-10),cv2.FONT_HERSHEY_PLAIN , 2, (0,255,0),2)
	cv2.imshow('img',image)
	print(result)
	if mode ==1:
		model.summary()
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	
	if len(sys.argv) > 1:
		if sys.argv[1].lower() == 'train':
			train()
		elif sys.argv[1] == 'detect' and len(sys.argv) == 3:
			predict(imgpath=sys.argv[2])
			
		elif sys.argv[1] == 'debug' and len(sys.argv) == 3:
			predict(imgpath=sys.argv[2],mode=1)
	else:
		
		print('\nuse main.py train to train')
		print('use main.py predict <img path> to detect and recognize digits')
		print('or use main.py debug <img path> to debug ')
