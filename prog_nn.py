#! /usr/bin/env python
# -*- coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

from param_collection import ParamCollection

#Helper functions.
def weight_variable(shape, stddev=0.1, initial=None):
    if initial is None:
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial)
    if initial is True:
        return tf.Variable(tf.ones(shape),trainable=False)

def bias_variable(shape, init_bias=0.1, initial=None):
    if initial is None:
        initial = tf.constant(init_bias, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)
    if initial is True:
        initial = tf.constant(init_bias, shape=shape, dtype=tf.float32)
        return tf.Variable(initial,trainable=False)
def adapters():
    a = tf.constant(1,shape=[1],dtype=tf.float32)
    return tf.Variable(a,trainable=True)

def get_checkpoint_path(base_path, column_i, checkpoint_i):
    print("base path is:", base_path)
    # Create file path here if non-existant?
    file_path = base_path+"/col"+str(column_i)
    file_name = base_path+"/col"+str(column_i)+"/checkpoint"+str(checkpoint_i)+".npy"
    print("file path is: ", file_name)
    # Verify path exists
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return file_path,file_name

class InitialColumnProgNN(object):
    """
    Descr: Initial network to train for later use transfer learning with a
        Progressive Neural Network.
    Args:
        n_input - The array length which the input image is flattened to
        kernel - A list of kernel size for each layer
        activations - A list of activation functions to use on the transforms.
        session - A TensorFlow session.
        checkpoing_base_path - Save path.
    Returns:
        None - attaches objects to class for InitialColumnProgNN.session.run()
    """

    #todo:add name to enery tensor
    def __init__(self, n_input, kernel,stride,activations, session, checkpoint_base_path, dtype=tf.float32):
        # Layers in network.
        self.session = session
        #self.L = len(topology)
        #self.topology = topology
        self.o_n = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
        self.imageIn = tf.reshape(self.o_n, shape=[-1,84,84,1])
        self.checkpoint_base_path = checkpoint_base_path

        self.W = []
        self.b =[]
        self.h = [self.imageIn]
        params = []

        padding = 'SAME'

        #The first two layers
        for k in range(2):
            #When training on second column, if the previous weights need to be frozen, set initial=True,
            # the variables are set as not trainable.
            self.W.append(weight_variable(kernel[k],initial=None))
            self.b.append(bias_variable([kernel[k][-1]],initial=None))
            conv = tf.nn.conv2d(self.h[-1],self.W[k],stride[k],padding)+ self.b[k]
            self.h.append(activations(conv))
            params.append(self.W[k])
            params.append(self.b[k])


        self.h.append(tf.layers.flatten(self.h[-1]))

        #fully connected layer
        self.W.append(weight_variable(kernel[-1],initial=None))
        self.b.append(bias_variable([kernel[-1][-1]],initial=None))
        fc = tf.matmul(self.h[-1], self.W[-1]) + self.b[-1]
        self.h.append(activations(fc))
        params.append(self.W[-1])
        params.append(self.b[-1])

        #Calculate value
        self.W.append(weight_variable([256,1],initial=None))
        self.b.append(bias_variable([1],initial=None))
        self.value = tf.matmul(self.h[-1], self.W[-1]) + self.b[-1]
        params.append(self.W[-1])
        params.append(self.b[-1])

        #Calculate policy
        self.W.append(weight_variable([256,6],initial=None))
        self.b.append(bias_variable([6],initial=None))
        fc = tf.matmul(self.h[-1], self.W[-1]) + self.b[-1]
        self.policy = tf.nn.softmax(fc)
        params.append(self.W[-1])
        params.append(self.b[-1])

        self.pc = ParamCollection(self.session, params)

    def add_input_to_feed_dict(self, feed_dict, input_batch):
        feed_dict[self.o_n] = input_batch
        return feed_dict

    def save(self, checkpoint_i):
        self.save_path,file_name = get_checkpoint_path(self.checkpoint_base_path, 0, checkpoint_i)
        current_params = self.pc.get_values_flat()
        np.save(file_name, current_params)

    def restore_weights(self, checkpoint_i):
        self.save_path, file_name = get_checkpoint_path(self.checkpoint_base_path, 0, checkpoint_i)
        saved_theta = np.load(file_name)
        self.pc.set_values_flat(saved_theta)





class ExtensibleColumnProgNN(object):
    """
    Descr: An extensible network column for use in transfer learning with a
        Progressive Neural Network.
    Args:
        n_input - The array length which the input image is flattened to
        kernel - A list of kernel size for each layer
        activations - A list of activation functions to use on the transforms.
        session - A TensorFlow session.
        checkpoing_base_path - Save path.
        prev_columns - Previously trained columns, either Initial or Extensible,
            we are going to create lateral connections to for the current column.
    Returns:
        None - attaches objects to class for ExtensibleColumnProgNN.session.run()
    """

    def __init__(self, n_input,kernel,stride, activations, session,checkpoint_base_path, prev_columns, dtype=tf.float32):
        self.session = session
        self.width = len(prev_columns)
        # Layers in network. First value is n_input, so it doesn't count.
        L = 5
        self.prev_columns = prev_columns
        self.checkpoint_base_path = checkpoint_base_path
        # Doesn't work if the columns aren't the same height
        #assert all([L == x.L for x in prev_columns])

        self.o_n = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
        self.imageIn = tf.reshape(self.o_n, shape=[-1,84,84,1])

        self.W = [[]] * L
        self.b = [[]] * L
        self.U = []
        self.V = []
        self.a = []
        for k in range(L-1):
            self.U.append( [[]] * self.width )
            self.V.append([[]]*self.width)
            self.a.append([[]]*self.width)
        self.h = [self.imageIn]  #h[0]
        # Collect parameters to hand off to ParamCollection.
        params = []
        padding = 'SAME'
        #first layer, not connected with previous layers
        self.W[0]=(weight_variable(kernel[0]))
        self.b[0]=(bias_variable([kernel[0][-1]]))
        conv = tf.nn.conv2d(self.h[-1], self.W[0], stride[0], padding) + self.b[0]
        self.h.append(activations(conv))  #h[1]
        params.append(self.W[0])
        params.append(self.b[0])

        #second layer
        self.W[1]=(weight_variable(kernel[1]))
        self.b[1]=(bias_variable([kernel[1][-1]]))
        preactivation = tf.nn.conv2d(self.h[-1], self.W[1], stride[1], padding) + self.b[1]
        for kk in range(self.width):
            self.a[0][kk] = adapters()
            ah = tf.multiply(self.a[0][kk],prev_columns[kk].h[1])
            maps_in= ah.get_shape().as_list()[3]
            maps_out = int(maps_in/(2.0*self.width))
            self.V[0][kk] = weight_variable([1,1,maps_in,maps_out])
            lateral = tf.nn.conv2d(ah,self.V[0][kk],stride[2],padding)
            lateral = activations(lateral)

            self.U[0][kk] = weight_variable([kernel[1][0],kernel[1][1],maps_out,kernel[1][3]])
            preactivation1 = tf.nn.conv2d(lateral, self.U[0][kk],stride[1],padding)
            preactivation = preactivation + preactivation1
        self.h.append(activations(preactivation))
        params.append(self.W[1])
        params.append(self.b[1])
        for kk in range(self.width):
            params.append(self.U[0][kk])
            params.append(self.V[0][kk])
            params.append(self.a[0][kk])

        self.h.append(tf.layers.flatten(self.h[-1]))  #h[3]

        #fully connected layer
        self.W[2]=(weight_variable(kernel[-1]))
        self.b[2]=(bias_variable([kernel[-1][-1]]))
        fc = tf.matmul(self.h[-1], self.W[2]) + self.b[2]
        for kk in range(self.width):
            self.a[1][kk] = adapters()
            ah = tf.multiply(self.a[1][kk],prev_columns[kk].h[2])
            maps_in = ah.get_shape().as_list()[3]
            maps_out = int(maps_in/(2.0*self.width))
            self.V[1][kk] = weight_variable([1,1,maps_in,maps_out])
            lateral = tf.nn.conv2d(ah,self.V[1][kk],stride[2],padding)
            lateral = activations(lateral)
            #lateral = tf.reshape(lateral,[-1,kernel[-1][-1]])
            lateral = tf.layers.flatten(lateral)
            self.U[1][kk] = weight_variable([lateral.get_shape().as_list()[-1],kernel[-1][-1]])
            fc += tf.matmul(lateral,self.U[1][kk])
        self.h.append(activations(fc))  #h[4]
        params.append(self.W[2])
        params.append(self.b[2])
        for kk in range(self.width):
            params.append(self.U[1][kk])
            params.append(self.V[1][kk])
            params.append(self.a[1][kk])

        #calculate value
        self.W[3] = (weight_variable([256,1]))
        self.b[3]=(bias_variable([1]))
        self.value = tf.matmul(self.h[-1], self.W[3]) + self.b[3]
        for kk in range(self.width):
            self.a[2][kk] = adapters()
            ah = tf.multiply(self.a[2][kk],prev_columns[kk].h[4])
            maps_in = ah.get_shape().as_list()[1]
            maps_out = int(maps_in/(2.0*self.width))
            self.V[2][kk] = weight_variable([maps_in,maps_out])
            lateral = tf.matmul(ah,self.V[2][kk])
            lateral = activations(lateral)

            self.U[2][kk] = weight_variable([maps_out,1])
            self.value += tf.matmul(lateral,self.U[2][kk])
        params.append(self.W[3])
        params.append(self.b[3])
        for kk in range(self.width):
            params.append(self.U[2][kk])
            params.append(self.V[2][kk])
            params.append(self.a[2][kk])

        #calculate policy
        self.W[4]=(weight_variable([256,6]))
        self.b[4]=(bias_variable([6]))
        fc = tf.matmul(self.h[-1], self.W[4]) + self.b[4]
        for kk in range(self.width):
            self.a[3][kk] = adapters()
            ah = tf.multiply(self.a[3][kk],prev_columns[kk].h[4])
            maps_in = ah.get_shape().as_list()[1]
            maps_out = int(maps_in/(2.0*self.width))
            self.V[3][kk] = weight_variable([maps_in,maps_out])
            lateral = tf.matmul(ah,self.V[3][kk])
            lateral = activations(lateral)

            self.U[3][kk] = weight_variable([maps_out,6])
            fc += tf.matmul(lateral,self.U[3][kk])
        self.policy = tf.nn.softmax(fc)
        params.append(self.W[4])
        params.append(self.b[4])
        for kk in range(self.width):
            params.append(self.U[3][kk])
            params.append(self.V[3][kk])
            params.append(self.a[3][kk])

        self.pc = ParamCollection(self.session, params)

    def add_input_to_feed_dict(self, feed_dict, input_batch):
        for col in self.prev_columns:
            feed_dict[col.o_n] = input_batch
        feed_dict[self.o_n] = input_batch
        return feed_dict

    def save(self, checkpoint_i):
        self.save_path, file_name= get_checkpoint_path(self.checkpoint_base_path,self.width , checkpoint_i)
        current_params = self.pc.get_values_flat()
        np.save(file_name, current_params)

    def restore_weights(self, checkpoint_i):
        self.save_path, file_name = get_checkpoint_path(self.checkpoint_base_path, self.width, checkpoint_i)
        saved_theta = np.load(file_name)
        self.pc.set_values_flat(saved_theta)