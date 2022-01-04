from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, UpSampling2D, concatenate
import tensorflow as tf
import json
import os

# load the params-model.json options
with open(os.path.join('v1', 'params-model.json')) as param_file:
    params_model = json.load(param_file)

regularizer = tf.keras.regularizers.L2(1e-4)

class ResNetLayer(Layer):
    def __init__(self, filters, downsample=False, **kwargs):
        super(ResNetLayer, self).__init__(**kwargs)
        self.downsample = downsample
        self.filters = filters
        self.bn_0 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_0')
        if self.downsample:
            self.conv_0_down = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_regularizer = regularizer,
                name='conv_0_down')
            self.conv_init_down = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                strides=2,
                padding='same',
                kernel_regularizer = regularizer,
                name='conv_0_down_init')
        else:
            self.conv_0_nodown = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_regularizer = regularizer,
                name='conv_0')
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_1')
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_regularizer = regularizer,
            name='conv_1')

    def call(self, inputs, training=True):
        x_inputs = inputs
        x = self.bn_0(x_inputs, training=training)
        x = tf.keras.activations.relu(x)
        if self.downsample:
            x = self.conv_0_down(x, training=training)
            x_inputs = self.conv_init_down(x_inputs, training=training)
        else:
            x = self.conv_0_nodown(x, training=training)
        x = self.bn_1(x,  training=training)
        x = tf.keras.activations.relu(x)
        x = self.conv_1(x, training=training)

        return tf.keras.layers.add([x, x_inputs])

class Classifier(Layer):
    def __init__(self, n_classes, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        #self.dropout = tf.keras.layers.Dropout(params_model['classifier']['dropout'], name='drop_0')
        self.conv_0 = tf.keras.layers.Conv2D(
                filters=n_classes,
                kernel_size=1,
                padding='same',
                #kernel_regularizer = regularizer,
                name='conv_0')

    def call(self, input, training=True):
        #x = self.dropout(input, training=training)
        x=input
        x = self.conv_0(x, training=training)
        return tf.keras.activations.softmax(x)

class FusionLayer(Layer):
    def __init__(self, type, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)
        self.recon_losses = []
        if type == 'sum':
            self.fusion = tf.keras.layers.Add()

        if type == 'concat':
            self.fusion = tf.keras.layers.Concatenate()

    def call(self, inputs, training):
        return self.fusion(inputs, training=training)

class UNET_Encoder(Layer):
    def __init__(self, filters, **kwargs):
        super(UNET_Encoder, self).__init__(**kwargs)
        self.filters = filters

        self.conv1 = Conv2D(filters[0], (3,3), activation='relu', padding='same', name = 'conv1')
        self.conv2 = Conv2D(filters[1], (3,3), activation='relu', padding='same', name = 'conv2')
        self.conv3 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv3')

        self.maxpool1 = MaxPool2D((2,2), name = 'maxPool1')
        self.maxpool2 = MaxPool2D((2,2), name = 'maxPool2')
        self.maxpool3 = MaxPool2D((2,2), name = 'maxPool3')

        self.conv4 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv4')
        self.conv5 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv5')
        self.conv6 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv6')

    def call(self, inputs, training):
        d1 = self.conv1(inputs, training = training) #128
        p1 = self.maxpool1(d1, training = training) #64

        d2 = self.conv2(p1, training = training) #64
        p2 = self.maxpool2(d2, training = training) #32

        d3 = self.conv3(p2, training = training) #32
        p3 = self.maxpool3(d3, training = training) #16

        b1 = self.conv4(p3, training = training) #16
        b2 = self.conv5(b1, training = training) #16
        b3 = self.conv6(b2, training = training) #16

        return b3, d3, d2, d1

class UNET_Decoder(Layer):
    def __init__(self, filters, n_classes, **kwargs):
        super(UNET_Decoder, self).__init__(**kwargs)
        self.filters = filters

        self.conv7 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv7')
        self.conv8 = Conv2D(filters[1], (3,3), activation='relu', padding='same', name = 'conv8')
        self.conv9 = Conv2D(filters[0], (3,3), activation='relu', padding='same', name = 'conv9')

        self.upsamp1 = UpSampling2D(size = (2,2), name = 'upSamp1')
        self.upsamp2 = UpSampling2D(size = (2,2), name = 'upSamp2')
        self.upsamp3 = UpSampling2D(size = (2,2), name = 'upSamp3')

        #self.last_conv = Conv2D(n_classes, (1,1), activation='softmax')


    def call(self, inputs, training):
        u3 = self.upsamp1(inputs[0], training = training) #32
        u3 = self.conv7(u3, training = training) #32
        m3 = concatenate([inputs[1], u3]) #32

        u2 = self.upsamp2(m3, training = training) #64
        u2 = self.conv8(u2, training = training) #64
        m2 = concatenate([inputs[2], u2]) #64

        u1 = self.upsamp3(m2, training = training) #128
        u1 = self.conv9(u1, training = training) #128
        m1 = concatenate([inputs[3], u1]) #128

        #out = self.last_conv(m1, training = training)

        return m1# out

class Conv2D_BN_RELU(Layer):
    def __init__(self, filters, padding = 'same', **kwargs):
        super(Conv2D_BN_RELU, self).__init__(**kwargs)
        self.conv = Conv2D(
            filters, 
            (3,3), 
            padding=padding, 
            kernel_regularizer=regularizer,
            name = 'conv')
        self.bn = tf.keras.layers.BatchNormalization(name='bn')

    def call(self, inputs, training):
        x = self.conv(inputs, training = training)
        x = self.bn(x, training=training)
        return tf.keras.activations.relu(x)

class CrossFusion(Layer):
    def __init__(self, filters, **kwargs):
        super(CrossFusion, self).__init__(**kwargs)
        self.h1 = Conv2D_BN_RELU(filters[0], name = 'h1')
        self.h2 = Conv2D_BN_RELU(filters[0], name = 'h2')
        self.j1 = Conv2D_BN_RELU(filters[1], name = 'j1')
        self.j2 = Conv2D_BN_RELU(filters[2], name = 'j2')

        self.j3 = tf.keras.layers.Conv2D(
                filters=params_model['classes'],
                kernel_size=1,
                padding='same',
                name='j3')

        self.recon_losses = []

    def call(self, inputs, training):
        x1_0 = inputs[0]
        x2_0 = inputs[1]

        x1 = self.h1(x1_0, training = training)
        x2 = self.h2(x2_0, training = training)
        x12 = self.h1(x2_0, training = training)
        x21 = self.h2(x1_0, training = training)

        j1 = concatenate((x1+x12, x2+x21))
        j2 = concatenate((x1, x21))
        j3 = concatenate((x12, x2))

        f1_0 = self.j1(j1, training = training)
        f2_0 = self.j1(j2, training = training)
        f3_0 = self.j1(j3, training = training)

        f1_1 = self.j2(f1_0, training = training)
        f2_1 = self.j2(f2_0, training = training)
        f3_1 = self.j2(f3_0, training = training)

        o1 = self.j3(f1_1, training = training)
        o2 = self.j3(f2_1, training = training)
        o3 = self.j3(f3_1, training = training)

        self.recon_losses = [
            tf.math.reduce_mean(tf.math.pow(o2-o1, 2)),
            tf.math.reduce_mean(tf.math.pow(o3-o1, 2))
            ]

        return tf.keras.activations.softmax(o1)

class ResNetBlock(Layer):
    def __init__(self, n_filters, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv_0 = Conv2D(n_filters, (3,3), activation='relu', padding='same', name = 'conv_0')
        self.conv_1 = Conv2D(n_filters, (3,3), activation='relu', padding='same', name = 'conv_1')
        self.conv_2 = Conv2D(n_filters, (3,3), activation='relu', padding='same', name = 'conv_2')

        self.dropout = tf.keras.layers.Dropout(0.5, name='dropout')

    def call(self, inputs, training):
        x_init = inputs

        x = self.conv_0(inputs, training = training)
        x = self.dropout(x, training=training)
        x = self.conv_1(x, training=training)

        x_init = self.conv_2(x_init, training=training)

        return tf.keras.layers.add([x, x_init])

class ResUNET_Encoder(Layer):
    def __init__(self, filters, **kwargs):
        super(ResUNET_Encoder, self).__init__(**kwargs)
        self.filters = filters

        self.conv1 = ResNetBlock(filters[0], name = 'resnet_block_0')
        self.conv2 = ResNetBlock(filters[1], name = 'resnet_block_1')
        self.conv3 = ResNetBlock(filters[2], name = 'resnet_block_2')

        self.maxpool1 = MaxPool2D((2,2), name = 'maxPool1')
        self.maxpool2 = MaxPool2D((2,2), name = 'maxPool2')
        self.maxpool3 = MaxPool2D((2,2), name = 'maxPool3')

        self.conv4 = ResNetBlock(filters[2], name = 'resnet_block_3')
        self.conv5 = ResNetBlock(filters[1], name = 'resnet_block_4')
        self.conv6 = ResNetBlock(filters[0], name = 'resnet_block_5')

    def call(self, inputs, training):
        d1 = self.conv1(inputs, training = training) #128
        p1 = self.maxpool1(d1, training = training) #64

        d2 = self.conv2(p1, training = training) #64
        p2 = self.maxpool2(d2, training = training) #32

        d3 = self.conv3(p2, training = training) #32
        p3 = self.maxpool3(d3, training = training) #16

        b1 = self.conv4(p3, training = training) #16
        b2 = self.conv5(b1, training = training) #16
        b3 = self.conv6(b2, training = training) #16

        return b3, d3, d2, d1

class ResUNET_Decoder(Layer):
    def __init__(self, filters, **kwargs):
        super(ResUNET_Decoder, self).__init__(**kwargs)
        self.filters = filters

        self.conv7 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv7')
        self.conv8 = Conv2D(filters[1], (3,3), activation='relu', padding='same', name = 'conv8')
        self.conv9 = Conv2D(filters[0], (3,3), activation='relu', padding='same', name = 'conv9')

        self.upsamp1 = UpSampling2D(size = (2,2), name = 'upSamp1')
        self.upsamp2 = UpSampling2D(size = (2,2), name = 'upSamp2')
        self.upsamp3 = UpSampling2D(size = (2,2), name = 'upSamp3')

        #self.last_conv = Conv2D(n_classes, (1,1), activation='softmax')


    def call(self, inputs, training):
        u3 = self.upsamp1(inputs[0], training = training) #32
        u3 = self.conv7(u3, training = training) #32
        m3 = concatenate([inputs[1], u3]) #32

        u2 = self.upsamp2(m3, training = training) #64
        u2 = self.conv8(u2, training = training) #64
        m2 = concatenate([inputs[2], u2]) #64

        u1 = self.upsamp3(m2, training = training) #128
        u1 = self.conv9(u1, training = training) #128
        m1 = concatenate([inputs[3], u1]) #128

        #out = self.last_conv(m1, training = training)

        return m1# out