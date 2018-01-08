import tensorflow as tf

def oneHot(x, depth):
    one = tf.Session().run(tf.one_hot(x, depth, 1, 0, -1))
    return one

def loadBatch():
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./data/dataset/training-images-07-0.tfrecords"))
	reader = tf.TFRecordReader()
	_, serialized = reader.read(filename_queue)

	features = tf.parse_single_example(serialized, features={'label':tf.FixedLenFeature([],tf.string), 'image':tf.FixedLenFeature([],tf.string)})

	record_image = tf.decode_raw(features['image'], tf.uint8)

	image = tf.reshape(record_image, [128, 128, 1])

	label = tf.cast(features['label'], tf.string)
	min_after_dequeue = 10000
	batch_size = 50
	capacity = 50000#min_after_dequeue + 5 * batch_size
	batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=4, capacity=capacity, min_after_dequeue=min_after_dequeue)

	with tf.Session() as sess:
		#init = (tf.global_variables_initializer(), tf.local_variables_initializer())
		#sess.run(init)
		sess.run(tf.local_variables_initializer())

		coord = tf.train.Coordinator()
		thread = tf.train.start_queue_runners(sess=sess,coord=coord)

		img, lab = sess.run(batch)
		image_batch = sess.run(tf.reshape(img, [-1, 16384]))
		label_batch = oneHot(lab, 33)

		coord.request_stop()
		coord.join(thread)

	return image_batch, label_batch

