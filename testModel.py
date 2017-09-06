import numpy as np
import sys
import os
import cv2
from keras.layers import Input, Convolution2D, Dense, UpSampling2D, PReLU, Dropout, \
    BatchNormalization, merge, Flatten
import keras.backend as K
from keras.models import Model
import shelve


def custom_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return -(2. * (K.sum(y_true_f * y_pred_f)) + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


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


def getIntersection(image, mask):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j] > 0 and image[i][j] > 10000:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    return mask


def get_test_data(path):
    patients = os.listdir(path)
    patients = list(set(patients) - {'._.DS_Store', '.DS_Store'})
    count = 0
    patient_dict = {}
    for i in range(len(patients)):
        patient = patients[i]
        slices = os.listdir(path + '/' + patient + '/' + 'pngs')
        count += len(slices)
        patient_dict[patient] = {}

    X = np.memmap("Xtest.dat", dtype=np.uint8, mode='w+', shape=(count, 1, 96, 96))
    arr_idx = 0
    for i in range(len(patients)):
        patient = patients[i]
        slices = os.listdir(path + '/' + patient + '/' + 'pngs')
        misc = os.listdir(path + '/' + patient + '/' + 'auxiliary')
        for j in range(len(slices)):
            img = cv2.imread(path + '/' + patient + '/' + 'pngs' + '/' + slices[j], flags=cv2.IMREAD_ANYDEPTH)
            img = (img / 256).astype('uint8')
            img = cv2.equalizeHist(img)
            img = img[110:390, 110:390]
            img = cv2.resize(img.copy(), (96, 96), interpolation=cv2.INTER_AREA)
            X[arr_idx + j][0] = np.array(img)
            slice_no = int(slices[j].split('.')[0])
            patient_dict[arr_idx + j] = patient + ',' + str(slice_no)
            patient_dict[patient][slice_no] = {}
            patient_dict[patient][slice_no]['idx'] = arr_idx + j
        arr_idx += len(slices)

        for j in range(len(misc)):
            slice_no = int(misc[j].split('.')[0])
            f = open(path + '/' + patient + '/' + 'auxiliary' + '/' + misc[j], mode='r')
            while (1):
                line = f.readline()
                if line == '':
                    break
                arr = line.strip().split(',')
                if arr[0] == "(0028.0030)":
                    patient_dict[patient][slice_no]['dx'] = float(arr[1])
                    patient_dict[patient][slice_no]['dy'] = float(arr[2])
                if arr[0] == "(0020.0032)":
                    patient_dict[patient][slice_no]['x0'] = float(arr[1])
                    patient_dict[patient][slice_no]['y0'] = float(arr[2])
        X.flush()
    aux_data = shelve.open('testCount.db')
    aux_data['count'] = count
    aux_data['test_dat'] = patient_dict


def prepare_submission(predictions, probabilities, patient_dict, path):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    with open('final.csv', 'w+') as f:
        for i in range(predictions.shape[0]):
            patient_slice = patient_dict[i].split(',')
            patient = patient_slice[0]
            slice_no = patient_slice[1]
            has_mask = (np.sum(predictions[i][0]) > 0)
            if probabilities[i][0] < 0.2:
                continue
            y = cv2.resize(predictions[i][0], (280, 280), interpolation=cv2.INTER_CUBIC)
            y = cv2.morphologyEx(y.copy(), cv2.MORPH_OPEN, kernel)
            yres = np.zeros((512, 512), dtype='uint8')
            yres[110:390, 110:390] = y
            # Get test image for fine tuning
            if has_mask:
                image_path = path + '/' + patient + '/pngs' + '/' + str(slice_no) + '.png'
                image = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
                cv2.normalize(image.copy(), image, 0, 65535, cv2.NORM_MINMAX)
                yres = getIntersection(image, yres.copy())
            #####
            _, cnts, _ = cv2.findContours(yres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            idx = 0
            largest_area = 0
            csv_line = patient_dict[i]
            contours = []
            for cnt in cnts:
                curr_area = cv2.contourArea(cnt)
                if curr_area >= largest_area:
                    largest_area = curr_area
                    csv_line = patient_dict[i]
                    contours = []
                else:
                    continue
                for point in cnt:
                    idx += 1
                    x1 = point[0][0]
                    y1 = point[0][1]
                    x0 = patient_dict[patient][slice_no]['x0']
                    y0 = patient_dict[patient][slice_no]['y0']
                    dx = patient_dict[patient][slice_no]['dx']
                    dy = patient_dict[patient][slice_no]['dy']
                    x = (x1 * dx) + x0
                    y = (y1 * dy) + y0
                    csv_line += ',' + str(x) + ',' + str(y)
                    contours.append([x, y])
            if len(contours) >= 5:
                f.write(csv_line + '\n')
    f.close()


def run(test_path):
    get_test_data(test_path)
    bs = 32
    aux_data = shelve.open('testCount.db')
    count = aux_data['count']
    patient_dict = aux_data['test_dat']
    Xtest = np.memmap("Xtest.dat", dtype=np.uint8, mode='r', shape=(count, 1, 96, 96))
    Xtest = Xtest.astype(np.float32)
    mu = np.load('mu.npy')
    std = np.load('std.npy')
    Xtest -= mu
    Xtest /= std
    model = keras_model(Xtest.shape[1:])
    model.load_weights(os.path.join('', 'model_weights.h5'))
    Ytest = model.predict(Xtest, batch_size=bs)
    predictions = Ytest[0]
    labels = Ytest[2]
    predictions = np.where(predictions >= 0.5, 255, 0)
    predictions = np.array(predictions, dtype='uint8')
    del Xtest
    prepare_submission(predictions, labels, patient_dict, test_path)


get_path = sys.argv[1]
run(get_path)
