#Handwritten Digit Recognation
handwritten digit recognation using OpenCV,sklearn,keras and Python
with CNN(Convolutional Neural Network) model

#Dependencies
1.Keras
2.cv2 (OpenCV)
3.numpy
4.pandas

#Contents
This repository contains the following files-
1.main.py       - Python Script to detect and recognise digits
2.model.h5      - Keras model weights(autogenerate after training)
3.model.yaml    - Keras model file (autogenerate after training)
4.test          - folder that contains test images
5.training-data - training data

#How to Use
For Linux- 

	python3 main.py detect imgpath - for detecting digits in image
	python3 main.py train          - for training the model
For Windows-

	python main.py detect imgpath - for detecting digits in image
	python main.py train          - for training the model
	
