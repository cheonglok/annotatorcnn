import os
import signal
import time
import shutil
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from .nets import get_network
from keras import optimizers, losses
import tensorflow as tf
from numpy import inf
from keras  import backend as K
from scipy.spatial.distance import directed_hausdorff, chebyshev

# force data format to "channels first"
keras.backend.set_image_data_format('channels_first')









def transform(Xb, yb):
    """
    handle class for on-the-fly data augmentation on batches.
    Applying 90,180 and 270 degrees rotations and flipping
    """
    # Flip a given percentage of the images at random:
    bs = Xb.shape[0]
    indices = np.random.choice(bs, bs // 2, replace=False)
    x_da = Xb[indices]

    # apply rotation to the input batch
    rotate_90 = x_da[:, :, :, ::-1, :].transpose(0, 1, 2, 4, 3)
    rotate_180 = rotate_90[:, :, :, :: -1, :].transpose(0, 1, 2, 4, 3)

    # apply flipped versions of rotated patches
    rotate_0_flipped = x_da[:, :, :, :, ::-1]
    rotate_180_flipped = rotate_180[:, :, :, :, ::-1]

    augmented_x = np.stack([rotate_180,
                            rotate_0_flipped,
                            rotate_180_flipped],
                            axis=1)

    # select random indices from computed transformations
    r_indices = np.random.randint(0, 3, size=augmented_x.shape[0])

    Xb[indices] = np.stack([augmented_x[i,
                                        r_indices[i], :, :, :, :]
                            for i in range(augmented_x.shape[0])])

    return Xb, yb


# def da_generator(x_train, y_train, batch_size=256):
#     """
#     Keras generator used for training with data augmentation. This generator
#     calls the data augmentation function yielding training samples
#     """
#     num_samples = x_train.shape[0]
#     while True:
#         for b in range(0, num_samples, batch_size):
#             x_ = x_train[b:b+batch_size]
#             y_ = y_train[b:b+batch_size]
#             x_, y_ = transform(x_, y_)
#             yield x_, y_

def da_generator(x_train, y_train, numsample, batch_size=256):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """
    num_samples = numsample



    # y_all = {}
    # x_all = {}
    while True:
        # for i in range (0,12):
        for b in range(0, num_samples, batch_size):
            # x_ = x_train[b:b + batch_size]
            x_ = [x_train[i][b:b + batch_size] for i in range(0, 5)]
            y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
            # y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
            # x_, y_ = transform([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            for i in range(0,5):
                x_[i], y_[i] = transform(x_[i], y_[i])

            # yield x_, y_
            yield ([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            # yield (x_all[0], [y_all[i] for i in range (0,12)])


def val_generator(x_train_v, y_train_v, numsample, batch_size=256):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """
    num_samples = numsample


    # y_all = {}
    # x_all = {}
    while True:
        # for i in range (0,12):
        for b in range(0, num_samples, batch_size):
            # x_ = x_train[b:b + batch_size]
            x_ = [x_train_v[i][b:b + batch_size] for i in range(0, 5)]
            y_ = [y_train_v[i][b:b + batch_size] for i in range(0, 5)]
            # y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
            # x_, y_ = transform([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            for i in range(0, 5):
                x_[i], y_[i] = transform(x_[i], y_[i])

            # yield x_, y_
            yield ([x_[i] for i in range(0, 5)], [y_[i] for i in range(0, 5)])
            # yield (x_all[0], [y_all[i] for i in range (0,12)])

# def val_generator(x_train, y_train, batch_size=256):
#     """
#     Keras generator used for training with data augmentation. This generator
#     calls the data augmentation function yielding training samples
#     """
#     num_samples = x_train.shape[0]
#     # y_all = {}
#     # x_all = {}
#     while True:
#         # for i in range (0,12):
#         for b in range(0, num_samples, batch_size):
#             x_ = x_train[b:b + batch_size]
#             y_ = [y_train[i][b:b + batch_size] for i in range(0, 5)]
#             x_, y_ = transform(x_, y_)
#             # yield x_, y_
#             yield (x_, [y_[i] for i in range(0, 5)])

def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    # np.random.shuffle(arr)
    return arr

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
def Jaccard_index(y_true, y_pred):
    smooth = 100.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    score = (intersection + smooth) / (union + smooth)
    return score


def Jaccard_loss(y_true, y_pred):
    loss = 1 - Jaccard_index(y_true, y_pred)
    return loss    

def true_false_positive_loss(y_true, y_pred, p_labels=0, label_smoothing=0, value=0):

    # y_pred_f = tf.reshape(y_pred, [-1]) 
    
    # y_pred_f = tf.reshape(y_pred, [-1])
    # this_size = K.int_shape(y_pred_f)[0]
    # # arr_len = this_size
    # # num_ones = 1
    # # arr = np.zeros(arr_len, dtype=int)
    # # if this_size is not None:
    # #      num_ones= np.int32((this_size * value * 100) / 100)
    # #      idx = np.random.choice(range(arr_len), num_ones, replace=False)
    # #      arr[idx] = 1
    # #      p_labels = arr
    # # else:
    # #      p_labels = np.random.randint(2, size=this_size)
    # if this_size is not None:
    #      p_labels = np.random.binomial(1, value, size=this_size)
    #      p_labels = tf.reshape(p_labels, y_pred.get_shape())
    # else: 
    #      p_labels =  y_pred 

    # p_lprint ('num_classes .....///', num_classes.shape[0])abels = tf.constant(y_pred_f)
    # print ('tf.size(y_pred_f) .....///', tf.size(y_pred_f))
    # num_classes = tf.dtypes.cast((tf.dtypes.cast(tf.size(y_pred_f), tf.int32) *  tf.dtypes.cast(value, tf.int32) * 100) / 100, tf.int32)
    
    # num_classes = 50
    # print ('num_classes .....', num_classes)
    # p_labels = tf.one_hot(tf.dtypes.cast(tf.zeros_like(y_pred_f), tf.int32), num_classes, 1, 0)
    # p_labels = tf.reduce_max(p_labels, 0)
    # p_labels = tf.reshape(p_labels, tf.shape(y_pred))
    # y_pred = K.constant(y_pred) if not tf.contrib.framework.is_tensor(y_pred) else y_pred
    # y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)

    C11 =  tf.math.multiply(y_true ,  y_pred)
    c_y_pred = 1 - y_pred
    C12 =  tf.math.multiply(y_true , c_y_pred)
    weighted_y_pred_u  =  tf.math.multiply(K.cast(p_labels, y_pred.dtype) , y_pred)
    weighted_y_pred_d  =  1 - weighted_y_pred_u
    y_pred = tf.math.add(tf.math.multiply(C11 , weighted_y_pred_u) , tf.math.multiply(C12 , weighted_y_pred_d))

    # y_pred /= tf.reduce_sum(y_pred,
    #                             reduction_indices=len(y_pred.get_shape()) - 1,
    #                             keep_dims=True)
    #     # manual computation of crossentropy
    # _EPSILON = 10e-8    
    # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # loss = - tf.reduce_sum(y_true * tf.log(y_pred),
    #                            reduction_indices=len(y_pred.get_shape()) - 1)

    loss = Jaccard_loss(y_true, y_pred)                           
    # with tf.GradientTape() as t:
    #     t.watch(y_pred)
    #     dpred = t.gradient(loss, y_pred)

    
    return loss



    # return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

def false_true_negative_loss(y_true, y_pred, p_labels=0, label_smoothing=0, value=0):

    
    # arr_len = this_size
    # num_ones = 1
    # arr = np.zeros(arr_len, dtype=int)
    # if this_size is not None:
    #      num_ones= np.int32((this_size * value * 100) / 100)
    #      idx = np.random.choice(range(arr_len), num_ones, replace=False)
    #      arr[idx] = 1
    #      p_labels = arr
    # else:
    #      p_labels = np.random.randint(2, size=this_size)
    # np.random.binomial(1, 0.34, size=10)
    

    # if this_size is not None:
    #     this_value= np.int32((this_size * value * 100) / 100)
    # else:
    #     this_value =  1
            
    # p_labels = np.random.randint(2, size=this_size)
    
    # p_labels = tf.constant(y_pred_f)
    # print ('tf.size(y_pred_f) .....///', tf.size(y_pred_f))
    # num_classes = tf.dtypes.cast((tf.dtypes.cast(tf.size(y_pred_f), tf.int32) *  tf.dtypes.cast(value, tf.int32) * 100) / 100, tf.int32)
    # num_classes = 50
    # print ('num_classes .....', num_classes)
    # p_labels = tf.one_hot(tf.dtypes.cast(tf.zeros_like(y_pred_f), tf.int32), num_classes, 1, 0)
    # # p_labels = tf.reduce_max(p_labels, 0)
    # p_labels = tf.reshape(p_labels, tf.shape(y_pred))
    # y_pred = K.constant(y_pred) if not tf.contrib.framework.is_tensor(y_pred) else y_pred
    # y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)


    
    c_y_true = 1 - y_true
    c_y_pred = 1 - y_pred
    C21 =  tf.math.multiply(c_y_true ,  y_pred)
    
    C22 =  tf.math.multiply(c_y_true , c_y_pred)
    weighted_y_pred_u  =  tf.math.multiply(K.cast(p_labels, y_pred.dtype) , y_pred)
    weighted_y_pred_d  =  1 - weighted_y_pred_u
    
    y_pred = tf.math.add(tf.math.multiply(C21 , weighted_y_pred_u) , tf.math.multiply(C22 , weighted_y_pred_d))
    # y_pred /= tf.reduce_sum(y_pred,
    #                             reduction_indices=len(y_pred.get_shape()) - 1,
    #                             keep_dims=True)
    #     # manual computation of crossentropy
    # _EPSILON = 10e-8  
    # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # loss = - tf.reduce_sum(y_true * tf.log(y_pred),
    #                            reduction_indices=len(y_pred.get_shape()) - 1)
    # with tf.GradientTape() as t:
    #     t.watch(y_pred)
    #     dpred = t.gradient(loss, y_pred)
    y_true = 1 - y_true
    loss = Jaccard_loss(y_true, y_pred) 
    
    return loss
def p_loss (y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    with tf.GradientTape() as t:
         t.watch(y_pred_f)
         pg = t.gradient(score, y_pred_f)
    return score, pg

def constrain(y_true, y_pred):
    loss, g = p_loss (y_true, y_pred)
    return loss

def constrain_loss(y_true, y_pred):
    return 1 - constrain(y_true, y_pred)


# def augmented_Lagrangian_loss(y_true, y_pred, augment_Lagrangian=1):

#     C_value, pgrad = p_loss(y_true, y_pred)
#     Ld, grad1 = loss_down(y_true, y_pred, from_logits=False, label_smoothing=0, value=C_value)
#     Lu, grad2 = loss_up(y_true, y_pred, from_logits=False, label_smoothing=0, value=C_value)
#     ploss = 1 - C_value
#     # adaptive lagrange multiplier 
#     _EPSILON = 10e-8
#     if all(v is not None for v in [grad1, grad2, pgrad]): 
#          alm = - ((grad1 + grad2) / pgrad + _EPSILON)
#     else:
#          alm =  augment_Lagrangian    
#     ploss = ploss * alm
#     total_loss = Ld + Lu + ploss
#     return total_loss

def calculate_gradient (y_true, y_pred, loss1, loss2):
    with tf.GradientTape(persistent=True) as t:
         t.watch(y_pred)
         g_loss1 = t.gradient(loss1, y_pred)
         g_loss2 = t.gradient(loss2, y_pred)
    loss, g_constrain = p_loss (y_true, y_pred)
    return   g_loss1, g_loss2, g_constrain    


def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg


def adaptive_lagrange_multiplier(loss1=None, loss2=None, loss3=None):
    _EPSILON = 10e-8
    augment_Lagrangian = 1
    if all(v is not None for v in [loss1, loss2, loss3]):
        res = ((loss1 + loss2) + _EPSILON) / (loss3 + _EPSILON)
        # print ("adaptive_lagrange_multiplier", r)
        return res
    else:
        # print("adaptive_lagrange_multiplier", augment_Lagrangian)
        return augment_Lagrangian


def total_loss(y_true, y_pred):
    # C_value = p_loss(y_true, y_pred)
    constrain_l = constrain_loss(y_true, y_pred)
    this_value = (-1 * constrain_l) + 1
    y_pred_f = tf.reshape(y_pred, [-1])
    this_size = K.int_shape(y_pred_f)[0]
    if this_size is not None:
        #  numpy.random.rand(4)
        #  p_labels = np.random.binomial(1, this_value, size=this_size)
        p_labels = rand_bin_array(this_value, this_size)
        p_labels = my_func(np.array(p_labels, dtype=np.float32))
        # p_labels = tf.dtypes.cast((tf.dtypes.cast(tf.size(p_labels), tf.int32)
        p_labels = tf.reshape(p_labels, y_pred.get_shape())

    else:
        p_labels = 0

    loss1 = true_false_positive_loss(y_true, y_pred, p_labels=p_labels, label_smoothing=0, value=this_value)
    loss2 = false_true_negative_loss(y_true, y_pred, p_labels=p_labels, label_smoothing=0, value=this_value)
    grad1, grad2, pgrad = calculate_gradient(y_true, y_pred, loss1, loss2)

    # ploss = 1 - C_value
    # adaptive lagrange multiplier
    # adaptive_lagrange_multiplier(y_true, y_pred):
    # _EPSILON = 10e-8
    # if all(v is not None for v in [grad1, grad2, pgrad]):
    #     return (((grad1 + grad2) + _EPSILON) / pgrad + _EPSILON)
    # else:
    #     return  augment_Lagrangian
    # ploss = ploss * alm
    lm = adaptive_lagrange_multiplier(grad1, grad2, pgrad)
    to_loss = loss1 + loss2 + (lm * constrain_l)
    return to_loss + losses.categorical_crossentropy(y_true, y_pred)


 ############################################################

# def total_loss(y_true, y_pred):
#     loss = losses.categorical_crossentropy(y_true, y_pred)
#     #loss = losses.categorical_crossentropy(y_true, y_pred)
#     return loss



def cascade_model(options):
    """
    3D cascade model using Nolearn and Lasagne

    Inputs:
    - model_options:
    - weights_path: path to where weights should be saved

    Output:
    - nets = list of NeuralNets (CNN1, CNN2)
    """

    # save model to disk to re-use it. Create an experiment folder
    # organize experiment
    if not os.path.exists(os.path.join(options['weight_paths'],
                                       options['experiment'])):
        os.mkdir(os.path.join(options['weight_paths'],
                              options['experiment']))
    if not os.path.exists(os.path.join(options['weight_paths'],
                                       options['experiment'], 'nets')):
        os.mkdir(os.path.join(options['weight_paths'],
                              options['experiment'], 'nets'))
    if options['debug']:
        if not os.path.exists(os.path.join(options['weight_paths'],
                                           options['experiment'],
                                           '.train')):
            os.mkdir(os.path.join(options['weight_paths'],
                                  options['experiment'],
                                  '.train'))

    # --------------------------------------------------
    # model 1
    # --------------------------------------------------
    all_loss = [total_loss, total_loss, total_loss, total_loss, total_loss]
    model = get_network(options)
    model.compile(loss=all_loss,
                  optimizer='adadelta',
                  metrics = ["accuracy"])

    # model = get_network(options)
    # model.compile(loss=total_loss,
    #               optimizer='adadelta',
    #               metrics=['accuracy', true_false_positive_loss , false_true_negative_loss, constrain_loss])
    # if options['debug']:
    #     model.summary()

    # save weights
    net_model_1 = 'model_1'
    net_weights_1 = os.path.join(options['weight_paths'],
                                options['experiment'],
                                'nets', net_model_1 + '.hdf5')

    net1 = {}
    net1['net'] = model
    net1['weights'] = net_weights_1
    net1['history'] = None
    net1['special_name_1'] = net_model_1

    # --------------------------------------------------
    # model 2
    # --------------------------------------------------

    model2 = get_network(options)
    model2.compile(loss=all_loss,
                   optimizer='adadelta',
                   metrics = ["accuracy"])
    # model2.compile(loss=total_loss,
    #               optimizer='adadelta',
    #               metrics=['accuracy', true_false_positive_loss , false_true_negative_loss, constrain_loss])
    # if options['debug']:
    #    model2.summary()

    # save weights
    # save weights
    net_model_2 = 'model_2'
    net_weights_2 = os.path.join(options['weight_paths'],
                                 options['experiment'],
                                 'nets', net_model_2 + '.hdf5')

    net2 = {}
    net2['net'] = model2
    net2['weights'] = net_weights_2
    net2['history'] = None
    net2['special_name_2'] = net_model_2

    # load predefined weights if transfer learning is selected

    if options['full_train'] is False:

        # load default weights
        print("> CNN: Loading pretrained weights from the", \
        options['pretrained_model'], "configuration")
        pretrained_model = os.path.join(options['weight_paths'], \
                                        options['pretrained_model'],'nets')
        model = os.path.join(options['weight_paths'],
                             options['experiment'])
        net1_w_def = os.path.join(model, 'nets', 'model_1.hdf5')
        net2_w_def = os.path.join(model, 'nets', 'model_2.hdf5')

        if not os.path.exists(model):
            shutil.copy(pretrained_model, model)
        else:
            shutil.copyfile(os.path.join(pretrained_model,
                                         'model_1.hdf5'),
                            net1_w_def)
            shutil.copyfile(os.path.join(pretrained_model,
                                         'model_2.hdf5'),
                            net2_w_def)
        try:
            net1['net'].load_weights(net1_w_def, by_name=True)
            net2['net'].load_weights(net2_w_def, by_name=True)
        except:
            print("> ERROR: The model", \
                options['experiment'],  \
                'selected does not contain a valid network model')
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    if options['load_weights'] is True:
        print("> CNN: loading weights from", \
            options['experiment'], 'configuration')
        print(net_weights_1)
        print(net_weights_2)

        net1['net'].load_weights(net_weights_1, by_name=True)
        net2['net'].load_weights(net_weights_2, by_name=True)

    return [net1, net2]


def define_training_layers(model, num_layers=1, number_of_samples=None):
    """
    Define the number of layers to train and freeze the rest

    inputs: - model: Neural network object net1 or net2 - number of
    layers to retrain - nunber of training samples

    outputs - updated model
    """
    # use the nunber of samples to choose the number of layers to retrain
    if number_of_samples is not None:
        if number_of_samples < 10000:
            num_layers = 1
        elif number_of_samples < 100000:
            num_layers = 2
        else:
            num_layers = 3

    # all layers are first set to non trainable

    net = model['net']
    for l in net.layers:
         l.trainable = False

    print("> CNN: re-training the last", num_layers, "layers")

    # re-train the FC layers based on the number of retrained
    # layers
    net.get_layer('out').trainable = True

    if num_layers == 1:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
    if num_layers == 2:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
        net.get_layer('dr_d2').trainable = True
        net.get_layer('d2').trainable = True
        net.get_layer('prelu_d2').trainable = True
    if num_layers == 3:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
        net.get_layer('dr_d2').trainable = True
        net.get_layer('d2').trainable = True
        net.get_layer('prelu_d2').trainable = True
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True

    #net.compile(loss='categorical_crossentropy',
    #            optimizer='adadelta',
    #            metrics=['accuracy'])

    model['net'] = net
    return model
def calcul_num_sample(x_train):
    temp = {}
    for label in range(0,5):
         temp[label] = x_train[label].shape[0]
         # print("temp[label]",label, temp[label])

    num_samples_t = min(temp.items(), key=lambda x: x[1])
    num_samples = num_samples_t[1]
    return num_samples

def fit_model(model, x_train, y_train, options, X_val, Y_val, initial_epoch=0):
    """
    fit the cascaded model.

    """
    num_epochs = options['max_epochs']
    train_split_perc = options['train_split']
    batch_size = options['batch_size']

    # convert labels to categorical
    # y_train = keras.utils.to_categorical(y_train, len(np.unique(y_train)))
    for label in range(0, 5):
        y_train[label] = keras.utils.to_categorical(y_train[label] == 1,
                                                    len(np.unique(y_train[label] == 1)))
        Y_val[label] = keras.utils.to_categorical(Y_val[label] == 1,
                                                    len(np.unique(Y_val[label] == 1)))

    train_val = {}
    train_val_v = {}
    # split training and validation
    for label in range(0, 5):
         perm_indices = np.random.permutation(x_train[label].shape[0])
         train_val[label] = int(len(perm_indices)*train_split_perc)
         perm_indices_v = np.random.permutation(X_val[label].shape[0])
         train_val_v[label] = int(len(perm_indices_v) * train_split_perc)

    x_train_ = {}
    y_train_ = {}
    x_val_ = {}
    y_val_ = {}

    train_val = max(train_val.items(), key=lambda x: x[1])
    train_val = train_val[1]
    train_val_v = max(train_val_v.items(), key=lambda x: x[1])
    train_val_v = train_val_v[1]

    # this = calcul_num_sample(x_train)
    # that = calcul_num_sample(X_val)

    for i in range(0, 5):
        x_train_[i] = x_train[i][:train_val]
        print("x_train_[", i, "]:", x_train_[i].shape[0])
        y_train_[i] = y_train[i][:train_val]
        print("y_train_[", i, "]:", y_train_[i].shape[0])
        x_val_[i] = X_val[i][:train_val_v]
        print("x_val_", i, "]:", x_val_[i].shape[0])
        y_val_[i] = Y_val[i][:train_val_v]
        print("y_val_[", i, "]:", y_val_[i].shape[0])

    numsamplex = calcul_num_sample(x_train_)
    numsamplexv = calcul_num_sample(x_val_)


    print ("num sample for training:", numsamplex)
    print("num sample for testing:", numsamplexv)





    h = model['net'].fit_generator(da_generator(
         x_train_, y_train_, numsamplex,
         batch_size=batch_size),
         validation_data=val_generator(x_val_, y_val_, numsamplexv, batch_size=batch_size),
         # validation_data = (x_val_, y_val_),
         epochs=num_epochs,
         initial_epoch=initial_epoch,
         steps_per_epoch=numsamplex/batch_size,
         validation_steps=numsamplexv/batch_size,

         verbose=options['net_verbose'],
         callbacks=[ModelCheckpoint(model['weights'],
                                     # monitor=['val_all_loss'],
                                     save_best_only=True,
                                     save_weights_only=True),
                     EarlyStopping(
                     # monitor='val_all_loss', in this case:  TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
                     #               min_delta=0,
                                    patience=options['patience']
                     #               verbose=0,
                     #               mode='auto')
                     )])
    # num_epochs = options['max_epochs']
    # train_split_perc = options['train_split']
    # batch_size = options['batch_size']
    #
    # # convert labels to categorical
    # # y_train = keras.utils.to_categorical(y_train, len(np.unique(y_train)))
    # y_train = keras.utils.to_categorical(y_train == 1,
    #                                      len(np.unique(y_train == 1)))
    #
    # # split training and validation
    # perm_indices = np.random.permutation(x_train.shape[0])
    # train_val = int(len(perm_indices)*train_split_perc)
    #
    # x_train_ = x_train[:train_val]
    # y_train_ = y_train[:train_val]
    # x_val_ = x_train[train_val:]
    # y_val_ = y_train[train_val:]
    #
    #
    #
    # h = model['net'].fit_generator(da_generator(
    #     x_train_, y_train_,
    #     batch_size=batch_size),
    #     validation_data=(x_val_, y_val_),
    #     epochs=num_epochs,
    #     initial_epoch=initial_epoch,
    #     steps_per_epoch=x_train_.shape[0]/batch_size,
    #     verbose=options['net_verbose'],
    #     callbacks=[ModelCheckpoint(model['weights'],
    #                                save_best_only=True,
    #                                save_weights_only=True),
    #                EarlyStopping(monitor='val_loss',
    #                              min_delta=0,
    #                              patience=options['patience'],
    #                              verbose=0,
    #                              mode='auto'),
    #                TensorBoard(log_dir='./tensorboardlogs',
    #                            histogram_freq=0,
    #                            #batch_size=32,
    #                            write_graph=True,
    #                            #write_grads=False,
    #                            #write_images=False,
    #                            #embeddings_freq=0,
    #                            #embeddings_layer_names=None,
    #                            #embeddings_metadata=None,
    #                            #embeddings_data=None,
    #                            #update_freq='epoch'
    #                            )],
    #
    #     max_q_size=np.int32(x_train_.shape[0]/batch_size),
    #     workers=4,
    #     pickle_safe=True)




    model['history'] = h

    if options['debug']:
        print("> DEBUG: loading best weights after training")

    model['net'].load_weights(model['weights'])

    return model

def fit_thismodel(model, x_train, y_train, options, initial_epoch=0):

    model['net'].load_weights(model['weights'])

    return model
