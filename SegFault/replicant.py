import numpy as np
import tensorflow as tf

inputShape = (16, 3, 64, 256, 6)
weightShape = (2, 16, 32, 6, 1028)

np_in = np.random.rand(*inputShape).astype(np.float32)
np_w = np.random.rand(*weightShape).astype(np.float32)

tf_in = tf.Variable(np_in)
tf_w = tf.Variable(np_w)

tf_out = tf.nn.conv3d(tf_in, tf_w, [1, 1, 1, 1, 1], padding="SAME")

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

np_out = sess.run(tf_out)
