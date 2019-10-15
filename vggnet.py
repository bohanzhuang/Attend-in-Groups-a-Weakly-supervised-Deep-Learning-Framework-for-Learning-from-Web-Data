
import lasagne
from lasagne.layers import LocalResponseNormalization2DLayer
from lasagne.layers import InputLayer, ReshapeLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import FeaturePoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.init import Normal
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from new_layers import Sumpool_bag_Layer, MeanpoolLayer, attention_layer, linear_sum_layer
import numpy as np
import theano
import theano.tensor as T


def build_train_model(batch_size, bag_size, input_var_train):

    net = {}
    net['input'] = InputLayer((batch_size*bag_size, 3, 224 ,224), input_var=input_var_train)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['attention'] = attention_layer(net['pool5'], W=Normal(0.001,0))
    net['linear_sum'] = linear_sum_layer([net['attention'], net['pool5']]) 
    net['mean_pool'] = MeanpoolLayer(net['linear_sum'])
    net['sum_pool'] = Sumpool_bag_Layer(net['mean_pool'], batch_size, bag_size)
    net['fc'] = DenseLayer(
        net['sum_pool'], num_units=431, W=lasagne.init.Normal(0.001), nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc'], softmax) 

    return net


def build_test_model(input_var_test):


    net = {}
    net['input'] = InputLayer((1, 3, 224 ,224), input_var=input_var_test)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['attention'] = attention_layer(net['pool5'], W=Normal(0.001,0))
    net['linear_sum'] = linear_sum_layer([net['attention'], net['pool5']]) 
    net['mean_pool'] = MeanpoolLayer(net['linear_sum'])
    net['fc'] = DenseLayer(
        net['mean_pool'], num_units=431, W=lasagne.init.Normal(0.01), nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc'], softmax)
    return net



def compile_train_model(config):


# build the training model
    train_batch_size = config['train_batch_size']   #number of bags
    bag_size = config['bag_size']
    input_var_train = T.tensor4('input_var_train')
    target_var = T.ivector('targets')
    train_network = build_train_model(train_batch_size, bag_size, input_var_train)

    learning_rate = theano.shared(np.float32(config['learning_rate']))
    classification_scores = lasagne.layers.get_output(train_network['prob'])
    debug_output = lasagne.layers.get_output(train_network['attention'])
    
    params = lasagne.layers.get_all_params(train_network['fc'], trainable=True)

    loss = T.mean(categorical_crossentropy(classification_scores, target_var))
    grads = T.grad(loss, params)

    for index, grad in enumerate(grads):
        if index > 25:
            grad *= 10.0

    y_pred = T.argmax(classification_scores, axis=1)
    error = T.mean(T.neq(y_pred, target_var))
    updates = lasagne.updates.nesterov_momentum(grads, params, learning_rate)

    train_model = theano.function([input_var_train, target_var], [loss, error], updates=updates) 
 
    return train_network, train_model, learning_rate


def compile_test_model(config):

    target_var = T.iscalar('targets')
    input_var_test = T.ftensor4('input_var_test')

    test_network = build_test_model(input_var_test)

    test_classification_scores = lasagne.layers.get_output(test_network['prob'], deterministic=True) 
    test_y_pred = T.argmax(test_classification_scores, axis=1)
    test_error = T.mean(T.neq(test_y_pred, target_var))

    test_model = theano.function([input_var_test, target_var], [test_error])

    return test_network, test_model



