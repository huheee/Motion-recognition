#디렉터리 미리 생성
import glob
import cv2
import numpy as np
import random


image_filenames = glob.glob("./image/right/*/*.jpg")

#random.shuffle(image_filenames)
for image in image_filenames:
	path = image.split("\\")

	img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
	#img = cv2.imread(image, cv2.IMREAD_COLOR)

	img = cv2.resize(img, (128, 128))
	#img = cv2.resize(img, (320, 240))

	#img = cv2.Canny(img, 100, 150)

	#kernel = np.ones((1,1), np.uint8)
	#img = cv2.dilate(img, kernel, iterations = 1)
	#img = cv2.erode(img, kernel, iterations=1)
	#img = 255 - img

	savepath = "./image/img07/" + path[1] + "/conv_" + path[2]

	cv2.imwrite(savepath, img)