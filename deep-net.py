#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
input > weights > hidden layer 1 (activation function) > 
weights > hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)

optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

Feed Forward + backprop = epoch
'''

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

#10 classes (0-9)

n_nodes_h1=500
n_nodes_h2=500
n_nodes_h3=500

n_classes =10
batch_size=100

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def model(data):
	hidden_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_h1])),
				'biases':tf.Variable(tf.random_normal([n_nodes_h1]))}
	hidden_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])),
				'biases':tf.Variable(tf.random_normal([n_nodes_h2]))}
	hidden_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])),
				'biases':tf.Variable(tf.random_normal([n_nodes_h3]))}
	output = {'weights':tf.Variable(tf.random_normal([n_nodes_h3, n_classes])),
				'biases':tf.Variable(tf.random_normal([n_classes]))}

	l_1 = tf.add(tf.matmul(data, hidden_1['weights']), hidden_1['biases'])
	l_1 = tf.nn.relu(l_1)

	l_2 = tf.add(tf.matmul(l_1, hidden_2['weights']), hidden_2['biases'])
	l_2 = tf.nn.relu(l_2)

	l_3 = tf.add(tf.matmul(l_2, hidden_3['weights']), hidden_3['biases'])
	l_3 = tf.nn.relu(l_3)

	output = tf.add(tf.matmul(l_3, output['weights']), output['biases'])

	return output

def train(x):
	prediction = model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#cycles of feed forward + backprop
	epochs = 10
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
				epoch_loss += c
			print('Epoch ', epoch, ' completed out of ', epochs, '. Loss: ', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y ,1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train(x)

