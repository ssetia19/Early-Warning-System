import os
import numpy as np   
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
import app.Utils as utils
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class ConvolutionalNeuralNetwork():

	def tflearn_definition(self):
        # Defination of Model 
		tf.reset_default_graph()
		convnet = input_data(shape=[None, utils.IMG_SIZE, utils.IMG_SIZE, 3], name='input')

		convnet = conv_2d(convnet, 32, 3, activation='relu')
		#convnet is the input layer
		#conv_2d is the convolution function
		#32 is 32x32 size of ? and 3 is ?
		#relu is activation function like sigmoid, tanh, relu=max(0,val), etc
		convnet = max_pool_2d(convnet, 2)

		convnet = conv_2d(convnet, 64, 3, activation='relu')
		convnet = max_pool_2d(convnet, 2)

		convnet = conv_2d(convnet, 128, 3, activation='relu')
		convnet = max_pool_2d(convnet, 2)

		convnet = conv_2d(convnet, 64, 3, activation='relu')
		convnet = max_pool_2d(convnet, 2)

		convnet = conv_2d(convnet, 32, 3, activation='relu')
		convnet = max_pool_2d(convnet, 2)

		#fully connected layer with relu activation
		convnet = fully_connected(convnet, 1024, activation='relu')
		convnet = dropout(convnet, 0.6)
		#dropout is regularization technique

		convnet = fully_connected(convnet, 512, activation='relu')
		convnet = dropout(convnet, 0.6)

		# fully connected with softmax activation ( OUTPUT LAYER )
		convnet = fully_connected(convnet, 10, activation='softmax')
		#10 is the final 10 nodes in the output layer because of 10 classes of output
		#ssoftmax makes the output between 0 and 1
		convnet = regression(convnet, optimizer='adam', learning_rate= utils.LR, loss='categorical_crossentropy', name='targets')
		#this is not linear regression type, it is actually ?
		#learning_rate decide by how much we update the weights during backpropagation in training period.
		

		return convnet
		
		

	def create_model(self , train_data ):

		train_data = np.load(train_data, encoding="latin1")
		train = train_data[:-7000] 
		validation = train_data[-7000:-1000] 

		x_train = np.array([i[0] for i in train]).reshape(-1, utils.IMG_SIZE, utils.IMG_SIZE, 3)
		y_train = [i[1] for i in train]

		x_validation = np.array([i[0] for i in validation]).reshape(-1, utils.IMG_SIZE, utils.IMG_SIZE, 3)
		y_validation = [i[1] for i in validation]

		convnet = self.tflearn_definition()
		model = tflearn.DNN(convnet, tensorboard_dir='./models/log', tensorboard_verbose=0)
		#DNN is deep neural network
		#verbose is ?

		if os.path.isfile("./models/cnn.model.meta"):
			model.load('./models/cnn.model')
		else:
			model.fit({'input': x_train}, {'targets': y_train}, n_epoch=10,#epoch is basically no of times we are gonno train it 
			          #when the whole training set passed through the ANN, that makes an EPOOCH
				  validation_set=({'input': x_validation}, {'targets': y_validation}), 
			          snapshot_step=500, show_metric=True, run_id=utils.MODEL_NAME)
			model.save('./models/cnn.model')

		return model


		
		
		

		
