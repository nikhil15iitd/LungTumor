import os
import cv2
import shelve
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D, MaxoutDense, Dense, UpSampling2D, PReLU, Dropout, \
    BatchNormalization, merge, LeakyReLU, ELU, Lambda, Flatten, AveragePooling2D
from keras.models import Model, model_from_yaml
from DataAug import ImageDataGenerator
import keras.backend as K
from keras.metrics import binary_crossentropy
from sklearn.cross_validation import KFold
from keras.optimizers import RMSprop

SEED = 97
BATCH_SIZE = 4000
crop_low = 100
crop_high = 400
RESIZE_IMAGE = crop_high - crop_low
ORIG_IMAGE = 512
smooth = 100.


def save_model(model, id):
    '''
    Save model architecture in JSON string format & model weights ( parameters )
    :param model: Keras Model
    :param id: Id of the Model
    :return:
    '''
    weights = 'model_weights' + str(id) + '.h5'
    model.save_weights(os.path.join('', weights), overwrite=True)


def load_model(model, id):
    '''
    Load Keras Model based on Id
    :param id: Model Id
    :return: Model
    '''
    weights = 'model_weights' + str(id) + '.h5'
    model.load_weights(os.path.join('', weights))
    return model


def bce(y_true, y_pred):
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)
    bce = binary_crossentropy(y_true_f, y_pred_f)
    return bce


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    val = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return val


def loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))


def np_dice_coef(y_true, y_pred):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    val = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return val


def inception_block(inputs, depth, batch_mode=0):
    # First Branch
    c1_1 = Convolution2D(depth / 4, 1, 1, init='he_normal', border_mode='same')(inputs)

    # Second Branch
    c2_1 = Convolution2D(depth / 8, 1, 1, init='he_normal', border_mode='same')(inputs)
    c2_1 = PReLU()(c2_1)
    c2_1 = Convolution2D(depth / 8, 3, 3, init='he_normal', border_mode='same')(c2_1)

    # Third Branch
    c3_1 = Convolution2D(depth / 8, 1, 1, init='he_normal', border_mode='same')(inputs)
    c3_1 = PReLU()(c3_1)

    c3_2 = Convolution2D(depth / 8, 1, 7, init='he_normal', border_mode='same')(c3_1)

    c3_3 = Convolution2D(depth / 8, 7, 1, init='he_normal', border_mode='same')(c3_1)

    c3_4 = Convolution2D(depth / 8, 5, 5, init='he_normal', border_mode='same')(c3_1)

    p4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inputs)
    c4_2 = Convolution2D(depth / 8, 1, 1, init='he_normal', border_mode='same')(p4_1)

    c4_3 = Convolution2D(depth / 8, 7, 7, init='he_normal', border_mode='same')(c4_2)

    res = merge([c1_1, c2_1, c3_2, c3_3, c3_4, c4_2, c4_3], mode='concat', concat_axis=1)
    res = BatchNormalization(mode=batch_mode, axis=1)(res)
    res = PReLU()(res)
    return res


def ConvolutionPooling2D(inputs, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
    l = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                      border_mode=border_mode)(inputs)
    l = BatchNormalization(mode=2, axis=1)(l)
    l = PReLU()(l)
    return l


def architecture(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = inception_block(inputs, 16, batch_mode=2)
    pool1 = ConvolutionPooling2D(conv1, 16, 3, 3, border_mode='same', subsample=(2, 2))
    pool1 = Dropout(0.4)(pool1)

    conv2 = inception_block(pool1, 32, batch_mode=2)
    pool2 = ConvolutionPooling2D(conv2, 32, 3, 3, border_mode='same', subsample=(2, 2))
    pool2 = Dropout(0.4)(pool2)

    conv3 = inception_block(pool2, 64, batch_mode=2)
    pool3 = ConvolutionPooling2D(conv3, 64, 3, 3, border_mode='same', subsample=(2, 2))
    pool3 = Dropout(0.4)(pool3)

    conv4 = inception_block(pool3, 128, batch_mode=2)
    pool4 = ConvolutionPooling2D(conv4, 128, 3, 3, border_mode='same', subsample=(2, 2))
    pool4 = Dropout(0.4)(pool4)

    conv5 = inception_block(pool4, 256, batch_mode=2)
    conv5 = Dropout(0.4)(conv5)

    aux = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid')(conv5)
    aux = Flatten()(aux)
    aux = Dense(256, activation='relu', init='he_normal')(aux)
    aux = Dropout(0.4)(aux)
    aux_out = Dense(1, activation='sigmoid', name='aux_out')(aux)

    up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up1, 128, batch_mode=2)
    conv6 = Dropout(0.4)(conv6)

    up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = inception_block(up2, 64, batch_mode=2)
    conv7 = Dropout(0.4)(conv7)

    out4x = Convolution2D(1, 1, 1, init='he_normal', border_mode='same', activation='sigmoid', name='out4x')(conv7)

    up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = inception_block(up3, 32, batch_mode=2)
    conv8 = Dropout(0.4)(conv8)

    out2x = Convolution2D(1, 1, 1, init='he_normal', border_mode='same', activation='sigmoid', name='out2x')(conv8)

    up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = inception_block(up4, 16, batch_mode=2)

    out = Convolution2D(1, 1, 1, init='he_normal', border_mode='same', activation='sigmoid', name='out')(conv9)

    model = Model(input=inputs, output=[out, out2x, out4x, aux_out])
    model.compile(optimizer='adam', loss={'out': loss, 'out2x': loss, 'out4x': loss,
                                          'aux_out': 'binary_crossentropy'},
                  metrics={'out': dice_coef},
                  loss_weights={'out': 1., 'out2x': 0.5, 'out4x': 0.25, 'aux_out': 0.5})

    return model


def writeContours(scanDict, sliceDict, auxiliary_test_data, ypred):
    file_name = 'submission_processed.csv'
    contoursData = {}
    with open(file_name, 'w+') as f:
        for i in range(ypred.shape[0]):
            scan_id = scanDict[i]
            slice_id = sliceDict[i]
            s = str(scan_id) + ',' + str(slice_id)
            resized_y = cv2.resize(ypred[i, 0], (RESIZE_IMAGE, RESIZE_IMAGE), interpolation=cv2.INTER_CUBIC)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            resized_y = cv2.morphologyEx(resized_y.copy(), cv2.MORPH_OPEN, kernel)
            # Pad with 0s since this label is result of cropping the original image
            padded = np.zeros((ORIG_IMAGE, ORIG_IMAGE), dtype=np.uint8)
            padded[crop_low:crop_high, crop_low:crop_high] = resized_y
            img, contours, hierarchy = cv2.findContours(padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            idx = 0
            for cnt in contours:
                for point in cnt:
                    idx += 1
                    x = point[0][0]
                    y = point[0][1]
                    x0 = auxiliary_test_data[scan_id][slice_id]['x0']
                    y0 = auxiliary_test_data[scan_id][slice_id]['y0']
                    dx = auxiliary_test_data[scan_id][slice_id]['dx']
                    dy = auxiliary_test_data[scan_id][slice_id]['dy']
                    xnew = (x * dx) + x0
                    ynew = (y * dy) + y0
                    s += ',' + str(xnew) + ',' + str(ynew)
            # Only write if there is some data
            if idx >= 2:
                if scan_id not in contoursData:
                    contoursData[scan_id] = {}
                    contoursData[scan_id][slice_id] = s
                elif slice_id not in contoursData[scan_id]:
                    contoursData[scan_id][slice_id] = s
                else:
                    contoursData[scan_id][slice_id] = s

        for key1 in contoursData.keys():
            keyView = contoursData[key1].keys()
            for key2 in keyView:
                if key2 + 1 in keyView or key2 - 1 in keyView or key2 + 2 in keyView or key2 - 2 in keyView:
                    f.write(contoursData[key1][key2] + '\n')
    f.close()


def get_auxiliary_output(mask_array):
    return np.array([int(np.sum(mask_array[i, 0]) > 0) for i in xrange(mask_array.shape[0])])


def downsample(mask_array, magnitude):
    result = np.zeros(
        (mask_array.shape[0], mask_array.shape[1], mask_array.shape[2] / magnitude, mask_array.shape[3] / magnitude),
        dtype=np.uint8)
    for i in range(mask_array.shape[0]):
        result[i][0] = cv2.resize(mask_array[i][0], (mask_array.shape[2] / magnitude, mask_array.shape[3] / magnitude),
                                  interpolation=cv2.INTER_AREA)
    return result


def main(num_epoch=10):
    datagen = ImageDataGenerator(
        rotation_range=5,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    X = np.load('train.npy')
    Y = np.load('mask_train.npy')
    print(Y.max())
    print(X.dtype)
    total_batches = X.shape[0] / BATCH_SIZE
    model = architecture(X.shape[1:])
    # model = load_model(model, 2)
    batches = 0
    for batch in datagen.flow(X, y=Y, batch_size=BATCH_SIZE, shuffle=True, seed=SEED):
        print('Batch no. ' + str(batches + 1))
        inputs, targets = batch
        targets2x = downsample(targets, 2)
        targets4x = downsample(targets, 4)
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)
        targets2x = targets2x.astype(np.float32)
        targets4x = targets4x.astype(np.float32)
        targets /= 255.
        targets2x /= 255.
        targets4x /= 255.
        targets_aux = get_auxiliary_output(targets)
        model.fit(inputs, [targets, targets2x, targets4x, targets_aux], batch_size=50, nb_epoch=1, shuffle=True)
        batches += 1
        if batches >= num_epoch * total_batches:
            break
    save_model(model, 2)
    del X
    del Y


main(num_epoch=35)
