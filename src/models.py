from keras.layers import *
from keras.regularizers import l2, l1_l2
from keras.models import Model						
import keras.backend as K								  
from classification_models.keras import Classifiers
import numpy as np


def relu_bn(x): 
  return Activation('relu')(BatchNormalization(axis=-1)(x))


def conv(x, nf, sz, wd, p, stride=1):
    x = Conv2D(nf, (sz, sz), strides=(stride, stride), padding='same', kernel_initializer='he_uniform',
               kernel_regularizer=l2(wd))(x)
    return Dropout(p)(x) if p else x


def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)


def dense_block(n, x, growth_rate, p, wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = Concatenate(axis=-1)([x, b])
        added.append(b)
    return x, added


def transition_dn(x, p, wd):
    # in the paper stride=1 but better results with stride=2
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)


def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = dense_block(n, x, growth_rate, p, wd)

        # keep track of skip connections
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added


def transition_up(added, wd=0):
    x = Concatenate(axis=-1)(added)
    _, r, c, ch = x.get_shape().as_list()
    return Conv2DTranspose(ch, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform',
                 kernel_regularizer = l2(wd))(x)


def up_path(added, skips, nb_layers, growth_rate, p, wd):
    # use previously saved list of skip connections
    for i, n in enumerate(nb_layers):
        x = transition_up(added, wd)

        # concatenate the skip connections
        x = Concatenate(axis=-1)([x, skips[i]])
        x, added = dense_block(n, x, growth_rate, p, wd)
    return x


def reverse(a): 
  return list(reversed(a))


def Densenet103(nb_classes, input_shape, nb_dense_block=6,
                    growth_rate=16, nb_filter=48, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4, weights_path=None):

    img_input = Input(shape=(input_shape))
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips, added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)
    _, r, c, f = x.get_shape().as_list()
    #x = Reshape((-1, nb_classes))(x)
    y = Activation('sigmoid')(x)
    
    model = Model(img_input, y)

    if(weights_path is not None):
        model.load_weights(weights_path)
    
    return model


def Resnet18(nb_classes, N, use_base_weights=True, weights_path=None, input_shape=(None,None,None,3)):
    if use_base_weights is True:
        base_weights = "imagenet"
    else:
        base_weights = None

    X = Input(shape=input_shape)

    x = Input(shape=input_shape[1:])
    ResNet18, _ = Classifiers.get('resnet18')
    ResNetModel = ResNet18(input_shape, input_tensor=x, weights=base_weights, include_top=False)
    
    res_out = ResNetModel.output
    out = GlobalAveragePooling2D(name='pool1')(res_out)
    out = Dense(nb_classes, name='fc1')(out)
    y = Dense(nb_classes, activation="softmax", name="predictions")(out)

    base_model = Model(inputs=x, outputs=y)

    for layer in base_model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(l1_l2(l1=1e-4, l2=1e-4)(layer.kernel))

        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(l1_l2(l1=1e-4, l2=1e-4)(layer.bias))

    preds = []
    for i in range(N):
        x = Lambda(lambda x: x[:, i, :, :, :])(X)
        y = base_model(x)
        y = Lambda(lambda x: K.expand_dims(x, axis=1))(y)
        preds.append(y)

    x2 = Lambda(lambda x: K.concatenate(x, axis=1))(preds)
    
    x2 = Lambda(lambda x: K.sum(x, axis=1))(x2)
    Y = Activation('softmax')(x2)

    model = Model(input=X, output=Y)

    if weights_path is not None:
        print(f"load model weights_path: {weights_path}")
        model.load_weights(weights_path)
    
    return model