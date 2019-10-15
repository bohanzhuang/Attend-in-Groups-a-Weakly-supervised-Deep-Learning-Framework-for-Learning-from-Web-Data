import numpy as np
import scipy.io as sio
import hickle as hkl
import os
import glob
import theano.tensor as T
import string
import scipy.misc
from random import shuffle


def proc_configs(weights_dir):
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print "Creat folder: " + weights_dir


def unpack_train_data(config, ext_data='.jpg', ext_label='.hkl'):


    train_img_dir = config['train_folder']
    sorted_train_dirs = sorted([name for name in os.listdir(train_img_dir)
                                if os.path.isdir(os.path.join(train_img_dir, name))])
    sorted_id_list = range(len(sorted_train_dirs))
    train_dict_wnid_to_sorted_id = {sorted_train_dirs[ind]: ind   
                                    for ind in sorted_id_list}    

    train_filenames = []

    for folder in sorted_train_dirs:
    	sub_train_filenames = []

        for name in os.listdir(os.path.join(train_img_dir, folder)):
            sub_train_filenames += [train_img_dir + folder + '/' + name]
        train_filenames.append(sorted(sub_train_filenames))

##----------------------------
    test_img_dir = config['val_folder']
    sorted_test_dirs = sorted([name for name in os.listdir(test_img_dir)
                                if os.path.isdir(os.path.join(test_img_dir, name))])
    sorted_id_list = range(len(sorted_test_dirs))
    test_dict_wnid_to_sorted_id = {sorted_test_dirs[ind]: ind   
                                    for ind in sorted_id_list}  


    test_filenames = []
    test_labels = []
    for folder in sorted_test_dirs:
        for name in os.listdir(os.path.join(test_img_dir, folder)):
            test_filenames += [test_img_dir + folder + '/' + name]
            test_labels += [test_dict_wnid_to_sorted_id[folder]]


    img_mean = np.load(config['mean_file'])
#    img_mean = np.rollaxis(img_mean, 2,0).astype('float32')
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')  # shape: 3*256*256*1

    return (train_filenames, test_filenames, test_labels, img_mean)


def unpack_test_data(config, ext_data='.hkl', ext_label='.hkl'):
    
    train_folder = config['train_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*' + ext_data)) 

    val_folder = config['val_folder']
    val_filenames = sorted(glob.glob(val_folder + '/*' + ext_data))  #.npy files

    img_mean = np.load(config['mean_file'])
    img_mean = np.rollaxis(img_mean, 2,0)
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')

    return (train_filenames, val_filenames, img_mean)


def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))   #learning_rate is the shared variable
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3][0] - val_record[-1][0] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx


def get_rand3d():
    tmp_rand = np.float32(np.random.rand(3))
    tmp_rand[2] = round(tmp_rand[2])
    return tmp_rand

def center_crop(data, param_rand, data_shape, cropsize=224):
    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = int(round(param_rand[0] * center_margin * 2))  # round: to the closest integer
    crop_ys = int(round(param_rand[1] * center_margin * 2))
    data = data[:, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]

    return np.asarray(data, dtype='float32')

def crop_and_mirror(data, param_rand, flag_batch=True, cropsize=224):
    '''
    when param_rand == (0.5, 0.5, 0), it means no randomness
    '''
    # print param_rand

    # if param_rand == (0.5, 0.5, 0), means no randomness and do validation
    # in training stage, use get_rand3d() to generate random variables
    if param_rand[0] == 0.5 and param_rand[1] == 0.5 and param_rand[2] == 0:
        flag_batch = True

    if flag_batch:
        # mirror and crop the whole batch
        crop_xs, crop_ys, flag_mirror = \
            get_params_crop_and_mirror(param_rand, data.shape, cropsize)

        # random mirror
        if flag_mirror:
            data = data[:, :, ::-1, :]

        # random crop
        data = data[:, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]

    else:
        # mirror and crop each batch individually
        # to ensure consistency, use the param_rand[1] as seed
        np.random.seed(int(10000 * param_rand[1]))

        data_out = np.zeros((data.shape[0], cropsize, cropsize,
                                data.shape[3])).astype('float32') #notice this form of definition

        for ind in range(data.shape[3]):
            # generate random numbers
            tmp_rand = np.float32(np.random.rand(3))
            tmp_rand[2] = round(tmp_rand[2])

            # get mirror/crop parameters
            crop_xs, crop_ys, flag_mirror = \
                get_params_crop_and_mirror(tmp_rand, data.shape, cropsize)

            # do image crop/mirror
            img = data[:, :, :, ind]
            if flag_mirror:
                img = img[:, :, ::-1]
            img = img[:, crop_xs:crop_xs + cropsize,
                      crop_ys:crop_ys + cropsize]
            data_out[:, :, :, ind] = img

        data = data_out

    return np.ascontiguousarray(data, dtype='float32')  #return a contiguous array in c01b order

def get_params_crop_and_mirror(param_rand, data_shape, cropsize):

    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = int(round(param_rand[0] * center_margin * 2))  # round: to the closest integer
    crop_ys = int(round(param_rand[1] * center_margin * 2))
    if False:
        # this is true then exactly replicate Ryan's code, in the batch case
        crop_xs = math.floor(param_rand[0] * center_margin * 2) # floor: the largest interger less/equal to x
        crop_ys = math.floor(param_rand[1] * center_margin * 2)

    flag_mirror = bool(round(param_rand[2]))

    return crop_xs, crop_ys, flag_mirror


def cross_entropy_loss(predict, target, config, scale):  # sign indicate whether general negative or not

    cost = -config['proportion'] * target * T.log(predict) - scale*(1-target) * T.log(1-predict + 1e-6)

    return cost


def unpack_test_configs(config, ext_data='.jpg'):
    label_dir = './preprocessed_data/ground_truth_labels.txt'
    with open(label_dir) as f:
        ground_truth_labels = f.readlines()
    for index in range(len(ground_truth_labels)):
        if string.find(ground_truth_labels[index], '\n') != -1:
            ground_truth_labels[index] = int(ground_truth_labels[index].strip('\n'))
        else:
            ground_truth_labels[index] = int(ground_truth_labels[index])
    train_filenames = sorted(glob.glob('./preprocessed_data/train/positive/' + '/*' + ext_data))
    return ground_truth_labels, train_filenames


def get_img(img_name, img_mean, img_size=256):

    target_shape = (img_size, img_size, 3)
    img = scipy.misc.imread(img_name)  # x*x*3
    assert img.dtype == 'uint8', img_name
    # assert False

    if len(img.shape) == 2:  #gray-scale image
        img = scipy.misc.imresize(img, (img_size, img_size))
        img = np.asarray([img, img, img])
        img = np.rollaxis(img, 0, 3)  #3*256*256
    else:
        if img.shape[2] > 3: #special image
            img = img[:, :, :3]
        img = scipy.misc.imresize(img, target_shape)

    img = img[:, :, ::-1, np.newaxis] - img_mean  
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)      
       
    return img  #3*256*256*1


def train_model_wrap(train_model, sampled_images,
                     batch_label, img_mean):   #give value to shared_x, shared_y, rand_arr


    feature_container = None

    for index in range(len(sampled_images)):
    	batch_feature = get_img(sampled_images[index], img_mean)
    	if feature_container is None:
    	    feature_container = batch_feature
    	else:
    	    feature_container = np.concatenate((feature_container, batch_feature), axis=3)

    param_rand = get_rand3d() 
    feature_container = crop_and_mirror(feature_container, param_rand, flag_batch=True, cropsize=224)
    feature_container = np.rollaxis(feature_container.astype('float32'), 3, 0)
    batch_label = np.asarray(batch_label).astype('int32')
    cost, error = train_model(feature_container, batch_label)

    return (cost, error)


def test_model_wrap(test_model, test_labels, test_filenames, img_mean):   

    test_features = []
    test_errors = []

    for index in xrange(len(test_filenames)):
        batch_feature = get_img(test_filenames[index], img_mean)
    	param_rand = np.float32([0.5, 0.5, 0])
    	batch_feature = center_crop(batch_feature, param_rand, batch_feature.shape)
        batch_feature = np.rollaxis(batch_feature.astype('float32'), 3, 0)
        batch_label = test_labels[index]
        error = test_model(batch_feature, batch_label)

        test_errors.append(error)

    this_test_error = np.mean(test_errors)

    return this_test_error

def get_detection_scores(scores_model, train_filenames, minibatch_index, img_mean):


    batch_feature = hkl.load(str(train_filenames[minibatch_index])) - img_mean
    param_rand = np.float32([0.5, 0.5, 0])
    batch_feature = center_crop(batch_feature, param_rand, batch_feature.shape)
    batch_feature = np.rollaxis(batch_feature.astype('float32'), 3, 0)
    scores = scores_model(batch_feature)
    return scores     



def debug_model_wrap(debug_model, minibatch_index, batch_size, train_filenames, img_mean):

    batch_feature = hkl.load(str(train_filenames[minibatch_index])) - img_mean
    param_rand = get_rand3d() 
    batch_feature = crop_and_mirror(batch_feature, param_rand, flag_batch=True, cropsize=224)
    batch_feature = np.rollaxis(batch_feature.astype('float32'), 3, 0)  
    output = debug_model(batch_feature)
    return output

