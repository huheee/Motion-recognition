import tensorflow as tf
import glob

image_filenames = glob.glob("./image/img07/*/*.jpg")

import cv2

from PIL import Image
import numpy as np

from itertools import groupby
from collections import defaultdict

training_dataset = defaultdict(list)
#testing_dataset = defaultdict(list)

image_filename_with_breed = map(lambda filename:(filename.split("\\")[1], filename), image_filenames)

for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x:x[0]):
	for i, breed_image in enumerate(breed_images):
		if i % 5 == 0:
			#testing_dataset[dog_breed].append(breed_image[1])
			training_dataset[dog_breed].append(breed_image[1])
		else:
			training_dataset[dog_breed].append(breed_image[1])

	breed_training_count = len(training_dataset[dog_breed])
	#breed_testing_count = len(testing_dataset[dog_breed])

	#assert round(breed_testing_count / (breed_training_count+breed_testing_count), 2) > 0.18, "Not enough testing images."


def write_records_file(dataset, record_location):
	#dataset:dict list
	#record_location:save path
	writer = None

	sess = tf.Session()

	current_index = 0
	for breed, images_filenames in dataset.items():
		for image_filename in images_filenames:
			print(image_filename, " index : " , current_index)
			if current_index % 2000 == 0:
				if writer:
					writer.close()
				record_filename = "{record_location}-{current_index}.tfrecords".format(record_location=record_location, current_index=current_index)

				writer = tf.python_io.TFRecordWriter(record_filename)
			
			current_index += 1

			image_file = tf.read_file(image_filename)

			try:
				image = tf.image.decode_jpeg(image_file)
			except:
				print(image_filename)
				continue

			#grayscale_image = tf.image.rgb_to_grayscale(image)
			size = tf.cast([128, 128], tf.int32)
			resized_image = tf.image.resize_images(image, size)
			image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
			image_label = breed.encode("utf-8")

			example = tf.train.Example(features=tf.train.Features(feature={'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])), 'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))}))

			writer.write(example.SerializeToString())
	writer.close()

#write_records_file(testing_dataset, "./dataset/testing-images")
write_records_file(training_dataset, "./dataset/training-images-07")