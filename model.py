from keras.models import Sequential,model_from_yaml
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D


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

def load_model():
	with open('model.yaml','r') as f:
		model_file = f.read()
	model = model_from_yaml(model_file)
	model.load_weights('model.h5')
	return model
