from sklearn.svm 			 import LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd
from model import *
import numpy as np
from keras.models import Sequential,model_from_yaml
from keras.layers import Dense,Dropout,Flatten
from keras.utils  import np_utils
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')


def load_data():
	df = pd.read_csv('./Training-data/train.csv')
	y = df['label']
	x = df.drop('label',1)
	y = np.array(y)
	x = np.array(x)
	
	train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2)
	del x,y 
	
	train_x = train_x.reshape(len(train_x),1,28,28).astype('float32')
	test_x  = test_x.reshape(len(test_x),1,28,28).astype('float32')

	train_x = train_x/255
	test_x  = test_x/255

	train_y = np_utils.to_categorical(train_y)
	test_y = np_utils.to_categorical(test_y)
	
	num_classes = test_y.shape[1]
	return train_x,test_x,train_y,test_y


def train():
	print('Loading data...',end='')
	train_x,test_x,train_y,test_y = load_data()
	print('Done')
	model = createmodel()
	print('Training...\n\n')
	model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=20,batch_size=200,verbose=2)
	print('\n\nTraning complete')
	print('testing on test data..',end='')
	scores = model.evaluate(test_x,test_y,verbose=0)
	print('Done')
	print('CNN Error:'+str(100-scores[1]*100))

	model_yaml = model.to_yaml()
	with open('model.yaml','w') as yaml_file:
		yaml_file.write(model_yaml)

	model.save_weights('model.h5')
	print('model saved')



