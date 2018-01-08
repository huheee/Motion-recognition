import os
import tensorflow as tf
import numpy as np
import cv2
import time
import ctypes

CWHITE  = '\33[37m'
RED = "\033[31m"
GREEN = '\033[92m'
YELLOW = '\033[93m'
END = '\033[0m'

#STD_OUTPUT_HANDLE_ID_ = ctypes.c_ulong(0xfffffff5)
#ctypes.windll.Kernel32.GetStdHandle.restype = ctypes.c_ulong
#COLOR_PRINT_ = ctypes.windll.Kernel32.GetStdHandle(STD_OUTPUT_HANDLE_ID_)

motion = [
"기본 자세\t\t", 
"오른팔 앞으로 내밀기\t", "오른팔 앞으로 내밀기\t", 
"오른팔 오른쪽으로 뻗기\t", "오른팔 오른쪽으로 뻗기\t", 
"오른팔 위로 뻗기\t", "오른팔 위로 뻗기\t", 
"오른팔 위로 굽히기\t","오른팔 위로 굽히기\t",
"오른팔 코 위로\t\t", "오른팔 코 위로\t\t", 
"오른손 머리 뒤에\t", "오른손 머리 뒤에\t", 
"오른손 왼쪽 무릎에\t", "오른손 왼쪽 무릎에\t", 
"오른손 허리 뒤로\t", "오른손 허리 뒤로\t", 
"왼팔 앞으로 내밀기\t","왼팔 앞으로 내밀기\t",
"왼팔 왼쪽으로 뻗기\t", "왼팔 왼쪽으로 뻗기\t", 
"왼팔 위로 뻗기\t\t", "왼팔 위로 뻗기\t\t", 
"왼팔 위로 굽히기\t", "왼팔 위로 굽히기\t", 
"왼팔 코 위로\t\t", "왼팔 코 위로\t\t", 
"왼손 머리 뒤에\t\t","왼손 머리 뒤에\t\t",
"왼손 오른쪽 무릎에\t","왼손 오른쪽 무릎에\t", 
"왼손 허리 뒤에\t\t","왼손 허리 뒤에\t\t"]

s0 = []
s1 = []
s2 = []

score = ["0점","2점","1점"]
#신경망 모델 구성
X = tf.placeholder(tf.float32, [None, 128, 128, 1])

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
#L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W4 = tf.Variable(tf.random_normal([32 * 32 * 128 , 512], stddev=0.01))
L4 = tf.reshape(L3, [-1, 32 * 32 * 128])
L4 = tf.matmul(L4, W4)
L4 = tf.nn.relu(L4)

W5 = tf.Variable(tf.random_normal([512, 33], stddev=0.01))
model = tf.matmul(L4, W5)
softmax = tf.nn.softmax(logits=model)

#신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(os.path.dirname("./train/"))
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess, ckpt.model_checkpoint_path)

img_num = 0

start = cv2.imread("./motion/li/0.jpg", 1)
cv2.imshow("motion", start)

while(1):
	asc = cv2.waitKey(1)

	if asc==44 or asc==46:#, .
		if asc==44:
			img_num -= 1
			if img_num < 1:
				img_num = 16

		if asc == 46:#.
			img_num += 1
			if img_num > 16:
				img_num = 1

		img = cv2.imread("./motion/li/"+str(img_num)+".jpg", 1)
		cv2.imshow("motion", img)

		img = cv2.resize(img, (128, 128))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gesture = sess.run(softmax, feed_dict={X:gray.reshape(-1, 128, 128, 1)}) * 100
		os.system('clear')
		#os.system('cls')

		max = 0
		for i in range(33):
			if gesture[0][i] > gesture[0][max]:
				max = i

		COLOR = CWHITE
		sc = 0
		for i in range(33):
			if i==0:
				sc = 0
			elif i%2==1:
				sc = 1
			else:
				sc = 2

			if i==max:
				COLOR = GREEN
				#ctypes.windll.Kernel32.SetConsoleTextAttribute(COLOR_PRINT_, 11)
			print(COLOR, "{}{} {:10.2f}%".format(motion[i], score[sc], gesture[0][i]), END)
			#print("{}{} {:10.2f}%".format(motion[i], score[sc], gesture[0][i]))
			COLOR = CWHITE	
			#ctypes.windll.Kernel32.SetConsoleTextAttribute(COLOR_PRINT_, 7)


os.system('clear')