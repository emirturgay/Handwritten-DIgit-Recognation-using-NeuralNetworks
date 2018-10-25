import sys
from preprocessing 	import *
from train 			import *
from predict 		import *

def loadimg(path):
	img = cv2.imread(path)
	return img


def main():
	# print(sys.argv[1])
	if(sys.argv[1]=="train"):
		train()
	elif(sys.argv[1] == "detect"):
		try:
			sys.argv[2]
		except Exception as e:
			print('use "python main.py decect <img_path>" to detect')
		else:
			path   = sys.argv[2]
			img    = loadimg(path)
			th_img = morphology(img)
			cv2.imshow('winname', th_img)
			# cv2.imshow('realimg', img)
			predict(img,th_img)
		# cv2.waitKey(0)
	else:
		print('\n\nuse "python main.py train" to train ')
		print('use "python main.py decect <img_path>" to detect')



if __name__ == '__main__':
	main()