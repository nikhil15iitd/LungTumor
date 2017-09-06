import numpy as np
import cv2
import os
import sys
import shelve

tumors = ['radiomics_gtv', 'Radiomics_gtv', 'radiomics_gtv2', 'radiomics_gtv_nw', 'radiomics_gtvr']


def get_train_data(path):
    patients = os.listdir(path)
    patients = list(set(patients) - {'._.DS_Store', '.DS_Store'})
    patients.sort()
    count = 0
    patient_dict = {}
    for i in range(len(patients)):
        patient = patients[i]
        patient_dict[patient] = {}
        slices = os.listdir(path + '/' + patient + '/' + 'pngs')
        count += len(slices)
        f = open(path + '/' + patient + '/' + 'structures.dat', mode='r')
        while (1):
            line = f.readline()
            if line == '':
                break
            arr = line.strip().split('|')
            for el in arr:
                if el in tumors:
                    patient_dict[patient]['tumor_index'] = arr.index(el) + 1

    X = np.memmap("Xtrain.dat", dtype=np.uint8, mode='w+', shape=(count, 1, 96, 96))
    Y = np.memmap("Ytrain.dat", dtype=np.uint8, mode='w+', shape=(count, 1, 96, 96))
    aux_data = shelve.open('trainCount.db')
    aux_data['count'] = count
    arr_idx = 0
    for i in range(len(patients)):
        patient = patients[i]
        slices = os.listdir(path + '/' + patient + '/' + 'pngs')
        contours = os.listdir(path + '/' + patient + '/' + 'contours')
        misc = os.listdir(path + '/' + patient + '/' + 'auxiliary')
        for j in range(len(slices)):
            slice_no = int(slices[j].split('.')[0])
            patient_dict[patient][slice_no] = {}
            patient_dict[patient][slice_no]['idx'] = arr_idx + j
            img = cv2.imread(path + '/' + patient + '/' + 'pngs' + '/' + slices[j], flags=cv2.IMREAD_ANYDEPTH)
            img = (img / 256).astype('uint8')
            img = cv2.equalizeHist(img)
            img = img[110:390, 110:390]
            img = cv2.resize(img.copy(), (96, 96), interpolation=cv2.INTER_AREA)
            X[arr_idx + j][0] = np.array(img)
            Y[arr_idx + j][0] = np.zeros(img.shape, dtype=np.uint8)
        arr_idx += len(slices)
        for j in range(len(misc)):
            slice_no = int(misc[j].split('.')[0])
            f = open(path + '/' + patient + '/' + 'auxiliary' + '/' + misc[j], mode='r')
            while (1):
                line = f.readline()
                if line == '':
                    break
                arr = line.strip().split(',')
                if arr[0] == '(0028.0030)':
                    patient_dict[patient][slice_no]['dx'] = float(arr[1])
                    patient_dict[patient][slice_no]['dy'] = float(arr[2])
                if arr[0] == '(0020.0032)':
                    patient_dict[patient][slice_no]['x0'] = float(arr[1])
                    patient_dict[patient][slice_no]['y0'] = float(arr[2])

        for j in range(len(contours)):
            slice_no = int(contours[j].split('.')[0])
            struct = int(contours[j].split('.')[1])
            if struct == patient_dict[patient]['tumor_index']:
                cnt = []
                mask = np.zeros((512, 512), dtype='uint8')
                f = open(path + '/' + patient + '/' + 'contours' + '/' + contours[j], mode='r')
                while (1):
                    line = f.readline()
                    if line == '':
                        break
                    arr = line.strip().split(',')
                    for k in range(0, len(arr), 3):
                        x = (float(arr[k]) - patient_dict[patient][slice_no]['x0']) / patient_dict[patient][slice_no][
                            'dx']
                        y = (float(arr[k + 1]) - patient_dict[patient][slice_no]['y0']) / \
                            patient_dict[patient][slice_no]['dy']
                        x = int(x)
                        y = int(y)
                        mask[y][x] = 255
                        cnt.append([x, y])
                cnt = np.array(cnt)
                mask = cv2.fillConvexPoly(mask, cnt, (255, 0, 0))
                mask = mask[110:390, 110:390]
                mask = cv2.resize(mask.copy(), (96, 96), interpolation=cv2.INTER_AREA)
                idx = patient_dict[patient][slice_no]['idx']
                Y[idx][0] = mask
        X.flush()
        Y.flush()


get_path = sys.argv[1]
get_train_data(get_path)
