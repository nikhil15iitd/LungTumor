import numpy as np
import os
import cv2
import shelve
from keras_image_generator import ImageDataGenerator
from keras.layers import Input, Convolution2D, Dense, UpSampling2D, PReLU, Dropout, \
    BatchNormalization, merge, Flatten
from keras.models import Model
import keras.backend as K

np.random.seed(123)


def custom_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return -(2. * (K.sum(y_true_f * y_pred_f)) + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def check_accuracy(y_true, y_pred):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    return (2. * np.sum(y_true_f * y_pred_f) + 1.) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.)


def resize(arr):
    res = np.zeros((arr.shape[0], 1, 48, 48))
    for i in range(arr.shape[0]):
        res[i][0] = cv2.resize(arr[i][0], (48, 48), interpolation=cv2.INTER_AREA)
    return res


def pool(inputs, nb_filter, nb_row, nb_col):
    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=(2, 2), border_mode='same')(inputs)
    x = BatchNormalization(mode=2, axis=1)(x)
    return PReLU()(x)


def keras_model(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Convolution2D(16, 3, 3, init='he_normal', border_mode='same')(inputs)
    conv1 = Convolution2D(16, 3, 3, init='he_normal', border_mode='same')(conv1)
    conv1 = BatchNormalization(mode=2, axis=1)(conv1)
    conv1 = PReLU()(conv1)
    pool1 = pool(conv1, 16, 3, 3)
    pool1 = Dropout(0.5)(pool1)

    conv2 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(pool1)
    conv2 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(conv2)
    conv2 = BatchNormalization(mode=2, axis=1)(conv2)
    conv2 = PReLU()(conv2)
    pool2 = pool(conv2, 32, 3, 3)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(pool2)
    conv3 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(conv3)
    conv3 = BatchNormalization(mode=2, axis=1)(conv3)
    conv3 = PReLU()(conv3)
    pool3 = pool(conv3, 64, 3, 3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(pool3)
    conv4 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(conv4)
    conv4 = BatchNormalization(mode=2, axis=1)(conv4)
    conv4 = PReLU()(conv4)
    pool4 = pool(conv4, 128, 3, 3)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(pool4)
    conv5 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(conv5)
    conv5 = BatchNormalization(mode=2, axis=1)(conv5)
    conv5 = PReLU()(conv5)
    conv5 = Dropout(0.5)(conv5)

    outbin = Convolution2D(256, 1, 1, init='he_normal')(conv5)
    outbin = Flatten()(outbin)
    outbin = Dense(1, activation='sigmoid', name='outbin')(outbin)

    up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(up1)
    conv6 = Convolution2D(128, 3, 3, init='he_normal', border_mode='same')(conv6)
    conv6 = BatchNormalization(mode=2, axis=1)(conv6)
    conv6 = PReLU()(conv6)
    conv6 = Dropout(0.5)(conv6)

    up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(up2)
    conv7 = Convolution2D(64, 3, 3, init='he_normal', border_mode='same')(conv7)
    conv7 = BatchNormalization(mode=2, axis=1)(conv7)
    conv7 = PReLU()(conv7)
    conv7 = Dropout(0.5)(conv7)

    up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(up3)
    conv8 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same')(conv8)
    conv8 = BatchNormalization(mode=2, axis=1)(conv8)
    conv8 = PReLU()(conv8)
    conv8 = Dropout(0.5)(conv8)

    out2 = Convolution2D(1, 1, 1, activation='sigmoid', name='out2')(conv8)

    up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(16, 3, 3, init='he_normal', border_mode='same')(up4)
    conv9 = Convolution2D(16, 3, 3, init='he_normal', border_mode='same')(conv9)
    conv9 = BatchNormalization(mode=2, axis=1)(conv9)
    conv9 = PReLU()(conv9)
    conv9 = Dropout(0.5)(conv9)

    out = Convolution2D(1, 1, 1, activation='sigmoid', name='out')(conv9)

    model = Model(input=inputs, output=[out, out2, outbin])
    model.compile(optimizer='adam', loss={'out': custom_loss, 'out2': custom_loss, 'outbin': 'binary_crossentropy'},
                  loss_weights={'out': 1., 'out2': 0.5, 'outbin': 0.5})
    return model


def run():
    num_epoch = 50
    bs = 32
    aux_data = shelve.open('trainCount.db')
    count = aux_data['count']
    X = np.memmap("Xtrain.dat", dtype=np.uint8, mode='r', shape=(count, 1, 96, 96))
    Y = np.memmap("Ytrain.dat", dtype=np.uint8, mode='r', shape=(count, 1, 96, 96))
    X = X[:].astype(np.float32)
    Y = Y[:].astype(np.float32)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    aux_data = shelve.open('testCount.db')
    np.save('mu.npy', mu)
    np.save('std.npy', std)
    aux_data['mean'] = std
    X -= mu
    X /= std
    Xval = X[-600:]
    Yval = Y[-600:]
    Yval /= 255.
    Xtrain = X[:-600]
    Ytrain = Y[:-600]
    del X
    del Y
    last_best = 0
    iterations_done = 0
    epochs = num_epoch * Xtrain.shape[0] / 5000
    model = keras_model(Xtrain.shape[1:])
    augmentor = ImageDataGenerator(rotation_range=5, shear_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
                                   horizontal_flip=True, vertical_flip=True)
    for batch in augmentor.flow(Xtrain, y=Ytrain, batch_size=5000, shuffle=True):
        inputs, targets = batch
        out2 = resize(targets)
        targets /= 255.
        out2 /= 255.
        outbin = np.array([int(np.sum(targets[i]) > 0) for i in range(targets.shape[0])])
        model.fit(inputs, [targets, out2, outbin], batch_size=bs, nb_epoch=2, shuffle=True)
        ypred = model.predict(Xval, batch_size=bs)[0]
        valid_score = check_accuracy(Yval, ypred)
        print('Dice Score on Validation Set: ' + str(valid_score))
        if valid_score > 0.60:
            model.save_weights(os.path.join('', 'model_weights.h5'), overwrite=True)
            last_best = valid_score
        iterations_done += 1
        if iterations_done > epochs:
            break
    if last_best == 0:
        model.save_weights(os.path.join('', 'model_weights.h5'), overwrite=True)


run()
