from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.advanced_activations import PReLU as prelu
from keras.layers.normalization import BatchNormalization as BN
from keras import backend as K
from keras import regularizers
from keras.models import Model
import keras
# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

def get_network(options):
    """
    CNN model for MS lesion segmentation. Based on the model proposed on:

    Valverde, S. et al (2017). Improving automated multiple sclerosis lesion
    segmentation with a cascaded 3D convolutional neural network approach.
    NeuroImage, 155, 159-168. https://doi.org/10.1016/j.neuroimage.2017.04.034

    However, two additional fully-connected layers are added to increase
    the effective transfer learning
    """

    # model options
    channels = len(options['modalities'])

    net_input1 = Input(name='in1', shape=(channels,) + options['patch_size'])
    net_input2 = Input(name='in2', shape=(channels,) + options['patch_size'])
    net_input3 = Input(name='in3', shape=(channels,) + options['patch_size'])
    net_input4 = Input(name='in4', shape=(channels,) + options['patch_size'])
    net_input5 = Input(name='in5', shape=(channels,) + options['patch_size'])


    merged = keras.layers.Concatenate(axis=1)([net_input1,net_input2,net_input3,net_input4,net_input5])

    # net_input = [net_input1,net_input2,net_input3,net_input4,net_input5]
    layer = Conv3D(filters=32, kernel_size=(3, 3, 3),
                   name='conv1_1',
                   activation=None,
                   padding="same")(merged)

    layer = BN(name='bn_1_1', axis=1)(layer)
    layer = prelu(name='prelu_conv1_1')(layer)
    layer = Conv3D(filters=32,
                   kernel_size=(3, 3, 3),
                   name='conv1_2',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_1_2', axis=1)(layer)
    layer = prelu(name='prelu_conv1_2')(layer)
    layer = MaxPooling3D(pool_size=(2, 2, 2),
                         strides=(2, 2, 2))(layer)
    layer = Conv3D(filters=64,
                   kernel_size=(3, 3, 3),
                   name='conv2_1',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_2_1', axis=1)(layer)
    layer = prelu(name='prelu_conv2_1')(layer)
    layer = Conv3D(filters=64,
                   kernel_size=(3, 3, 3),
                   name='conv2_2',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_2_2', axis=1)(layer)
    layer = prelu(name='prelu_conv2_2')(layer)
    layer = MaxPooling3D(pool_size=(2, 2, 2),
                         strides=(2, 2, 2))(layer)
    layer = Flatten()(layer)
    layer = Dropout(name='dr_d1', rate=0.5)(layer)
    layer = Dense(units=256,  activation=None, name='d1')(layer)
    layer = prelu(name='prelu_d1')(layer)
    layer = Dropout(name='dr_d2', rate=0.5)(layer)
    layer = Dense(units=128,  activation=None, name='d2')(layer)
    layer = prelu(name='prelu_d2')(layer)
    layer = Dropout(name='dr_d3', rate=0.5)(layer)
    layer = Dense(units=64,  activation=None, name='d3')(layer)
    layer = prelu(name='prelu_d3')(layer)
    # net_output = Dense(units=2, name='out', activation='softmax')(layer)

    output1 = Dense(units=2, name='output_label1', activation='softmax')(layer)
    output2 = Dense(units=2, name='output_label2', activation='softmax')(layer)
    output3 = Dense(units=2, name='output_label3', activation='softmax')(layer)
    output4 = Dense(units=2, name='output_label4', activation='softmax')(layer)
    output5 = Dense(units=2, name='output_label5', activation='softmax')(layer)



    # model = Model(inp,[output1,output2,output3,output4,output5])

    # output1 = Dense(1, activation = 'sigmoid')(x)
    # output2 = Dense(1, activation = 'sigmoid')(x)
    # output3 = Dense(1, activation = 'sigmoid')(x)
    # output4 = Dense(1, activation = 'sigmoid')(x)
    # output5 = Dense(1, activation = 'sigmoid')(x)


    # net_output = [output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12]
    model = Model(inputs=[net_input1,net_input2,net_input3,net_input4,net_input5], outputs=[output1,output2,output3,output4,output5])




    # model = Model(inputs=[net_input], outputs=net_output)

    return model
