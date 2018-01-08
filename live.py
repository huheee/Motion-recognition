import os
import tensorflow as tf
import numpy as np
import cv2
import time

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

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while(1):
	ret, frame = cap.read()
	cv2.imshow('frame', frame)

	#if int(time.time()%3)==0:
	if cv2.waitKey(1) & 0xFF == ord('c'):
		img = cv2.resize(frame, (128, 128))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gesture = sess.run(softmax, feed_dict={X:gray.reshape(-1, 128, 128, 1)}) * 100
		os.system('cls')

		print("{}{:10.4f} {}".format('기본 자세\t\t\t', gesture[0][0], '0점'))
		print("오른팔 앞으로 내밀기", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][1], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][2], '1점'))
		print("오른팔 오른쪽으로 뻗기", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][3], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][4], '1점'))
		print("오른팔 위로 뻗기", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][5], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][6], '1점'))
		print("오른팔 위로 굽히기", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][7], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][8], '1점'))
		print("오른팔 코 위로  ", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][9], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][10], '1점'))
		print("오른손 머리 뒤에", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][11], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][12], '1점'))
		print("오른손 왼쪽 무릎에", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][13], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][14], '1점'))
		print("오른손 허리 뒤로", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][15], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][16], '1점'))
		print("왼팔 앞으로 내밀기", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][17], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][18], '1점'))
		print("왼팔 오른쪽으로 뻗기", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][19], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][20], '1점'))
		print("왼팔 위로 뻗기   ", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][21], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][22], '1점'))
		print("왼팔 위로 굽히기", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][23], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][24], '1점'))
		print("왼팔 코 위로     ", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][25], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][26], '1점'))
		print("왼손 머리 뒤에   ", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][27], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][28], '1점'))
		print("왼손 오른쪽 무릎에", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][29], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][30], '1점'))
		print("왼손 허리 뒤에   ", end='\t\t')
		print("{:10.4f} {}".format(gesture[0][31], '2점'), end='\t')
		print("{:10.4f} {}".format(gesture[0][32], '1점'))
