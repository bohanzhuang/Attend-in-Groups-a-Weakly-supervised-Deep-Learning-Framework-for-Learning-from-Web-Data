import scipy.io as sio
import sys
import time
import os
import yaml
import numpy as np
import glob
import math
import hickle as hkl
import random


from utils import train_model_wrap, test_model_wrap, proc_configs, unpack_train_data, adjust_learning_rate

def main(config):

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(config['gpu'])
    import lasagne
    from vggnet import compile_train_model, compile_test_model

    (train_filenames, test_filenames, test_labels, img_mean) = unpack_train_data(config) 
    (train_network, train_model, learning_rate) = compile_train_model(config)
    (test_network, test_model) = compile_test_model(config)


    test_batch_size = config['test_batch_size']
    train_batch_size = config['train_batch_size']
    bag_size = config['bag_size']
    n_iters = config['n_iters']



    print("Compilation complete, starting training...")   

    epoch = 0
    step_idx = 0
    test_record = []

    model_file = config['finetune_weights_dir'] 
    weights = np.load(model_file)
    lasagne.layers.set_all_param_values(train_network['conv5_3'], weights[0:26])

    while epoch < config['n_epochs']:

    	epoch = epoch + 1
        if config['resume_train'] and epoch == 1:
            load_epoch = config['load_epoch']
            model_file = './weights/model_' + str(load_epoch) + '.npy'
            weights = np.load(model_file)
            lasagne.layers.set_all_param_values(train_network['prob'], weights)
            epoch = load_epoch + 1
            lr_to_load = np.load(
                config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
            test_record = list(
                np.load(config['weights_dir'] + 'test_record.npy'))
            learning_rate.set_value(lr_to_load)

        count = 0

        for iteration in range(n_iters):

            num_iter = (epoch - 1) * n_iters + count
            count = count + 1
            if count == 1:
                s = time.time()
            if count == 20:
                e = time.time()
                print "time per 20 iter:", (e - s)

# random sample
            batch_img_names = []
            batch_label = []
            for index in range(train_batch_size):
                class_index = random.choice(range(431))
                class_names = train_filenames[class_index]
                sampled_img_names = [class_names[i] for i in random.sample(xrange(len(class_names)), bag_size)]
                batch_img_names += sampled_img_names
                batch_label.append(class_index)

            cost, error = train_model_wrap(train_model, batch_img_names, batch_label, img_mean)


            if num_iter % config['print_freq'] == 0:
                print 'training @ iter = ', num_iter
                print 'training cost:', cost
                if config['print_train_error']:
                    print 'training error rate:', error             


        step_idx = adjust_learning_rate(config, epoch, step_idx,
                                            test_record, learning_rate)

        if epoch % config['snapshot_freq'] == 0:
            model_file = config['weights_dir'] + 'model_' + str(epoch) + '.npy'
            values = lasagne.layers.get_all_param_values(train_network['fc'])
            np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                       learning_rate.get_value())
            np.save(model_file, values)

### transfer
        train_params = lasagne.layers.get_all_param_values(train_network['fc'])
        lasagne.layers.set_all_param_values(test_network['fc'], train_params)

        this_test_error = test_model_wrap(test_model, test_labels, test_filenames, img_mean)

        print('epoch %i: validation error %f %%' %
              (epoch, this_test_error * 100.))
        test_record.append([this_test_error])

        np.save(config['weights_dir'] + 'test_record.npy', test_record)
 




if __name__ == '__main__':

    with open('./config.yaml', 'r') as f:
        config = yaml.load(f)
        
    proc_configs(config['weights_dir'])
    main(config)















