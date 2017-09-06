import os
import cv2
import shelve
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D, MaxoutDense, Dense, UpSampling2D, PReLU, Dropout, \
    BatchNormalization, merge, LeakyReLU, ELU, Lambda, Flatten, AveragePooling2D, ZeroPadding2D
from keras.models import Model, model_from_yaml
from DataAug import ImageDataGenerator
from keras.metrics import binary_crossentropy
from keras.layers import Layer, InputSpec
import keras.initializations
import keras.backend as K

SEED = 23
BATCH_SIZE = 1024
crop_low = 122
crop_high = 378
RESIZE_IMAGE = crop_high - crop_low
ORIG_IMAGE = 512
smooth = 100.
NFOLDS = 3
train_dir = 'C:\\Users\\nikhil\\LungTumor\\example_extracted\\example_extracted'
test_dir = 'C:\\Users\\nikhil\\LungTumor\\provisional_extracted_no_gt\\provisional_extracted_no_gt'


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


def iterate_minibatches_train(inputs, targets, indices, batchsize):
    for start_idx in range(0, indices.shape[0] - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def inception_block(inputs, depth, batch_mode=0):
    # First Branch
    c1_1 = Convolution2D(depth / 4, 1, 1, init='he_normal', border_mode='same')(inputs)
    c1_1 = BatchNormalization(mode=batch_mode, axis=1)(c1_1)
    c1_1 = PReLU()(c1_1)

    # Second Branch
    c2_1 = Convolution2D(depth / 4, 3, 3, init='he_normal', border_mode='same')(c1_1)

    # Third Branch
    c3_2 = Convolution2D(depth / 8, 1, 7, init='he_normal', border_mode='same')(c1_1)

    c3_3 = Convolution2D(depth / 8, 7, 1, init='he_normal', border_mode='same')(c3_2)

    c3_4 = Convolution2D(depth / 8, 5, 1, init='he_normal', border_mode='same')(c1_1)

    c3_4 = Convolution2D(depth / 8, 1, 5, init='he_normal', border_mode='same')(c3_4)

    p4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inputs)
    c4_2 = Convolution2D(depth / 4, 1, 1, init='he_normal', border_mode='same')(p4_1)

    res = merge([c1_1, c2_1, c3_3, c3_4, c4_2], mode='concat', concat_axis=1)
    res = ELU()(res)
    return res


def ConvolutionPooling2D(inputs, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
    l = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                      border_mode=border_mode)(inputs)
    l = BatchNormalization(mode=2, axis=1)(l)
    l = PReLU()(l)
    return l


def BatchConvolution2D(inputs, nb_filter, nb_row, nb_col, border_mode='same'):
    l = Convolution2D(nb_filter, nb_row, nb_col, border_mode=border_mode, bias=False)(inputs)
    l = BatchNormalization(mode=2)(l)
    l = PReLU()(l)
    return l


def architecture2x(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = inception_block(inputs, 32, batch_mode=2)
    pool1 = ConvolutionPooling2D(conv1, 32, 3, 3, border_mode='same', subsample=(2, 2))
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
    aux_out = Dense(1, activation='sigmoid', name='aux_out')(aux)

    up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up1, 64, batch_mode=2)
    conv6 = Dropout(0.4)(conv6)

    up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = inception_block(up2, 32, batch_mode=2)
    conv7 = Dropout(0.4)(conv7)

    out4x = Convolution2D(1, 1, 1, init='he_normal', border_mode='same', activation='sigmoid', name='out4x')(conv7)

    up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = inception_block(up3, 32, batch_mode=2)
    conv8 = Dropout(0.4)(conv8)

    out2x = Convolution2D(1, 1, 1, init='he_normal', border_mode='same', activation='sigmoid', name='out2x')(conv8)

    up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = inception_block(up4, 16, batch_mode=2)
    conv9 = Dropout(0.25)(conv9)

    out = Convolution2D(1, 1, 1, init='he_normal', border_mode='same', activation='sigmoid', name='out')(conv9)

    model = Model(input=inputs, output=[out, out2x, out4x, aux_out])
    model.compile(optimizer='adam', loss={'out': loss, 'out2x': loss, 'out4x': loss,
                                          'aux_out': 'binary_crossentropy'},
                  metrics={'out': dice_coef},
                  loss_weights={'out': 1., 'out2x': 0.2, 'out4x': 0.05, 'aux_out': 0.01})

    return model


class Maxout2D(Layer):
    def __init__(self, output_dim, cardinality, init='glorot_uniform', **kwargs):
        super(Maxout2D, self).__init__(**kwargs)
        # the k of the maxout paper
        self.cardinality = cardinality
        # the m of the maxout paper
        self.output_dim = output_dim
        self.init = keras.initializations.get(init)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_shape[1], input_shape[2], input_shape[3]))]
        self.W = self.init((self.input_dim, self.output_dim, self.cardinality),
                           name='{}_W'.format(self.name))
        self.b = K.zeros((self.output_dim, self.cardinality))
        self.trainable_weights = [self.W, self.b]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        # flatten the spatial dimensions
        flat_x = K.reshape(x, (-1, input_shape[1], input_shape[2] * input_shape[3]))
        output = K.dot(
            K.permute_dimensions(flat_x, (0, 2, 1)),
            K.permute_dimensions(self.W, (1, 0, 2))
        )
        output += K.reshape(self.b, (1, 1, self.output_dim, self.cardinality))
        output = K.max(output, axis=3)
        output = output.transpose(0, 2, 1)
        output = K.reshape(output, (-1, self.output_dim, input_shape[2], input_shape[3]))
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim, input_shape[2], input_shape[3])

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'cardinality': self.cardinality
        }
        base_config = super(Maxout2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def fcn(input_shape):
    input_img = Input(shape=input_shape, name='input_img')
    x = Convolution2D(32, 3, 3, border_mode='same')(input_img)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(32, 2)(x)
    pool1 = x = MaxPooling2D((2, 2), name='pool1')(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(32, 2)(x)
    pool2 = x = MaxPooling2D((2, 2), name='pool2')(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Maxout2D(64, 2)(x)
    pool3 = x = MaxPooling2D((2, 2), name='pool3')(x)

    # -- binary presence part
    x = Dropout(0.25)(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Maxout2D(64, 2)(x)
    pool4 = x = MaxPooling2D((2, 2), name='pool4')(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Maxout2D(64, 2)(x)
    pool5 = x = MaxPooling2D((2, 2), name='pool5')(x)

    # Since some images have not mask, the hope is that the innermost units capture this
    x = Flatten()(pool5)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dense(16)(x)
    x = LeakyReLU()(x)
    outbin = Dense(1, activation='sigmoid', name='outbin')(x)

    x = Maxout2D(32, 2)(pool5)
    outmap5 = Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid', name='outmap5')(x)
    x = UpSampling2D((2, 2))(x)

    x = merge([x, pool4], mode='concat', concat_axis=1)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(32, 2)(x)
    outmap4 = Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid', name='outmap4')(x)

    x = UpSampling2D((2, 2))(x)

    x = merge([x, pool3], mode='concat', concat_axis=1)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(16, 2)(x)
    x = UpSampling2D((2, 2))(x)

    x = merge([x, pool2], mode='concat', concat_axis=1)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(16, 2)(x)
    x = UpSampling2D((2, 2))(x)

    x = merge([x, pool1], mode='concat', concat_axis=1)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(16, 2)(x)
    x = UpSampling2D((2, 2))(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    outmap = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name='outmap')(x)

    model = Model(
        input=input_img,
        output=[outmap, outmap4, outmap5, outbin]
    )

    metrics = {'outmap': dice_coef}
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  loss_weights=[1., 0.01, 0.01, 0.05], metrics=metrics)
    return model


def writeContours(scanDict, sliceDict, auxiliary_test_data, ypred):
    file_name = 'submission_processed.csv'
    contoursData = {}
    with open(file_name, 'w+') as f:
        for i in range(ypred.shape[0]):
            scan_id = scanDict[i]
            slice_id = sliceDict[i]
            s = str(scan_id) + ',' + str(slice_id)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            resized_y = cv2.morphologyEx(ypred[i, 0].copy(), cv2.MORPH_OPEN, kernel)
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


def count_images(path):
    id1 = 0
    for scan in os.listdir(path):
        if scan == '._.DS_Store' or scan == '.DS_Store':
            continue
        scan_path = path + '\\' + scan
        slice_path = scan_path + '\\pngs'
        id2 = 0
        for image in os.listdir(slice_path):
            id2 += 1
        id1 += id2
    return id1


def main(num_epoch=10):
    bs = 16
    datagen = ImageDataGenerator(
        rotation_range=5,
        shear_range=0.08,
        width_shift_range=0.08,
        height_shift_range=0.08,
        fill_mode='nearest')
    new_size = crop_high - crop_low
    # '''
    train_samples = count_images(train_dir)
    X = np.memmap("Xtrain.dat", dtype=np.uint16, mode='r', shape=(train_samples, 1, new_size, new_size))
    Y = np.memmap("Ytrain.dat", dtype=np.uint8, mode='r', shape=(train_samples, 1, new_size, new_size))
    print(Y.max())
    print(X.dtype)
    print(X.shape)
    train_idxs = [i for i in range(X.shape[0])]
    train_idxs = np.array(train_idxs)
    Xval = X[train_idxs[-700:]].astype(np.float32)
    Yval = Y[train_idxs[-700:]].astype(np.float32)
    Yval /= 255.
    model = fcn(X.shape[1:])
    train_idxs = train_idxs[:-700]
    # model = load_model(model, 10)
    for epoch in range(num_epoch):
        batches = 0
        np.random.shuffle(train_idxs)
        for god_batch in iterate_minibatches_train(X, Y, train_idxs, BATCH_SIZE):
            Xtrain, Ytrain = god_batch
            for batch in datagen.flow(Xtrain, y=Ytrain, batch_size=BATCH_SIZE, shuffle=True, seed=SEED + batches):
                batches += 1
                print('Batch no. ' + str(batches))
                inputs, targets = batch
                targets2x = downsample(targets, 16)
                targets4x = downsample(targets, 32)
                inputs = inputs.astype(np.float32)
                targets = targets.astype(np.float32)
                targets2x = targets2x.astype(np.float32)
                targets4x = targets4x.astype(np.float32)
                targets /= 255.
                targets2x /= 255.
                targets4x /= 255.
                targets_aux = get_auxiliary_output(targets)
                inputs /= 65535.
                model.fit(inputs, [targets, targets2x, targets4x, targets_aux], batch_size=bs, nb_epoch=2, shuffle=True)
                break
            if batches % 3 == 0:
                yval_pred = model.predict(Xval, batch_size=bs)[0]
                print('Dice Score on Validation Set: ' + str(np_dice_coef(Yval, yval_pred)))
    del Xval
    del Yval
    save_model(model, 11)
    # '''
    testsamples = count_images(test_dir)
    Xtest = np.memmap("Xtest.dat", dtype=np.uint16, mode='r', shape=(testsamples, 1, new_size, new_size))
    scan_name = np.load('scan_test.npy')
    slice_id = np.load('slice_test_id.npy')
    myShelvedDict = shelve.open("test_auxiliary.db")
    auxiliary_test_data = myShelvedDict["auxiliary_test_data"]
    Y = np.memmap("Ytest.dat", dtype=np.uint8, mode='w+', shape=(testsamples, 1, new_size, new_size))
    model = architecture2x(Xtest.shape[1:])
    model = load_model(model, 11)
    for i in range(0, Xtest.shape[0] - BATCH_SIZE + 1, BATCH_SIZE):
        inputs = Xtest[i:i + BATCH_SIZE].astype(np.float32)
        ypred = model.predict(inputs, batch_size=bs)[0]
        ypred = np.where(ypred >= 0.5, 255, 0)
        ypred = ypred.astype(np.uint8)
        if i + BATCH_SIZE > Y.shape[0]:
            Y[i:] = ypred
        else:
            Y[i:i + BATCH_SIZE] = ypred
        Y.flush()
    print(Y.shape)
    writeContours(scan_name, slice_id, auxiliary_test_data, Y)


main(num_epoch=5)
