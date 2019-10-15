import lasagne
from lasagne.layers import Layer, MergeLayer
import numpy as np
import theano
import theano.tensor as T
import numpy as np
from lasagne.nonlinearities import softmax



class MeanpoolLayer(Layer):

    def get_output_for(self, input, **kwargs):
    	output = T.sum(T.sum(input, axis=3), axis=2)
    	return output

    def get_output_shape_for(self, input_shape):
    	return (input_shape[0], input_shape[1])


class Sumpool_bag_Layer(Layer):

    def __init__(self, incoming, batch_size, bag_size, **kwargs):
        super(Sumpool_bag_Layer, self).__init__(incoming, **kwargs)
        self.batch_size = batch_size
        self.bag_size = bag_size

    def get_output_for(self, input, **kwargs):
        input_reshape = T.reshape(input, (self.batch_size, self.bag_size, self.input_shape[1]))
        output = T.mean(input_reshape, axis=1)
        return output

    def get_output_shape_for(self, input_shape):
        return (self.batch_size, input_shape[1])   




class Maxpool_bag_Layer(Layer):

    def __init__(self, incoming, batch_size, bag_size, **kwargs):
        super(MaxpoolLayer, self).__init__(incoming, **kwargs)
        self.batch_size = batch_size
        self.bag_size = bag_size

    def get_output_for(self, input, **kwargs):
        input_reshape = T.reshape(input, (self.batch_size, self.bag_size, self.input_shape[1]))
        output = T.max(input_reshape, axis=1)
        return output

    def get_output_shape_for(self, input_shape):
        return (self.batch_size, input_shape[1])    


class RelupoolLayer(Layer):
    def __init__(self, incoming, batch_size, bag_size, **kwargs):
        super(RelupoolLayer, self).__init__(incoming, **kwargs)
        self.batch_size = batch_size
        self.bag_size = bag_size	
    def get_output_for(self, input, **kwargs):
        Relu_input = T.reshape(input, (self.batch_size, self.bag_size, self.input_shape[1]))       
        Relu_output = T.nnet.relu(Relu_input)
        output = T.sum(Relu_output, axis=1)
        return output
    def get_output_shape_for(self, input_shape):
        return (self.batch_size, input_shape[1]) 


class attention_layer(Layer):
    def __init__(self, incoming, W=lasagne.init.Normal(0.01), **kwargs):
        super(attention_layer, self).__init__(incoming, **kwargs)
        self.W = self.add_param(W, (self.input_shape[1], 1), name = 'W')
        
    def get_output_for(self, input, **kwargs):
        input_reshape = input.dimshuffle(0,2,3,1)
	output = T.extra_ops.squeeze(T.dot(input_reshape, self.W))
        self.output = output
        output_reshape = T.reshape(output, (self.input_shape[0], self.input_shape[2]*self.input_shape[3]))
        normalize_output = (T.nnet.nnet.softplus(output_reshape) + 0.1) / T.sum(T.nnet.nnet.softplus(output_reshape) + 0.1, axis=1).dimshuffle(0,'x')
        normalize_output_2 = T.reshape(normalize_output, (self.input_shape[0], self.input_shape[2], self.input_shape[3]))
	return normalize_output_2

    def get_output_shape_for(self, input_shape):
    	return (self.input_shape[0], self.input_shape[2], self.input_shape[3])

class linear_sum_layer(MergeLayer):

    def get_output_for(self, inputs, **kwargs):

	weights = T.extra_ops.repeat(inputs[0].dimshuffle(0,'x',1,2), self.input_shapes[1][1], axis=1)
	output = weights * inputs[1]
	return output

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]
	
