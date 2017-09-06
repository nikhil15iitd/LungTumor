import os
import numpy as np
import cv2
import shelve

train_dir = 'C:\\Users\\nikhil\\LungTumor\\example_extracted\\example_extracted'
test_dir = 'C:\\Users\\nikhil\\LungTumor\\provisional_extracted_no_gt\\provisional_extracted_no_gt'
ORIG_IMAGE = 512
RESIZE_IMAGE = 128
scan_properties = {}
# Check the index of tumor structure for each patient (or scan)
structure_variables = {}

tumor_names = ["radiomics_gtv", "radiomics_gtv", "Radiomics_gtv", "radiomics_gtv2", "radiomics_gtv_nw",
               "radiomics_gtvr"]
tumor_names = set(tumor_names)

crop_low = 122
crop_high = 378

train_shape = []
test_shape = []


def count_train_images(path, isTrain=True):
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
    new_size = crop_high - crop_low
    print(id1)
    if isTrain:
        X = np.memmap("Xtrain.dat", dtype=np.uint16, mode='w+', shape=(id1, 1, new_size, new_size))
        Y = np.memmap("Ytrain.dat", dtype=np.uint8, mode='w+', shape=(id1, 1, new_size, new_size))
        return X, Y
    else:
        X = np.memmap("Xtest.dat", dtype=np.uint16, mode='w+', shape=(id1, 1, new_size, new_size))
        return X


def create_train(path, X, Y):
    id1 = 0
    for scan in os.listdir(path):
        if scan == '._.DS_Store' or scan == '.DS_Store':
            continue
        scan_path = path + '\\' + scan
        scan_properties[scan] = {}
        with open(scan_path + '\\structures.dat', mode='r') as f:
            for line in f:
                curLine = line.split("|")
                for j in range(len(curLine)):
                    if curLine[j].strip() in tumor_names:
                        print(scan + ': ' + curLine[j].strip())
                        structure_variables[scan] = j + 1
                        break

        slice_path = scan_path + '\\pngs'
        id2 = 0
        for image in os.listdir(slice_path):
            scan_properties[scan][int(image.split(".")[0])] = {}
            scan_properties[scan][int(image.split(".")[0])]['index'] = id1 + id2
            image_path = slice_path + '\\' + image
            img = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
            img = img[crop_low:crop_high, crop_low:crop_high]
            cv2.normalize(img.copy(), img, 0, 65535, cv2.NORM_MINMAX)
            img_mask = np.zeros(img.shape, dtype=np.uint8)
            X[id1 + id2, 0] = np.array(img)
            Y[id1 + id2, 0] = np.array(img_mask)
            id2 += 1

        auxiliary_path = scan_path + '\\auxiliary'
        for aux in os.listdir(auxiliary_path):
            file_path = auxiliary_path + '\\' + aux
            image = int(aux.split(".")[0])
            with open(file_path, mode='r') as ax:
                for line in ax:
                    curLine = line.split(",")
                    if curLine[0] == "(0020.0032)":
                        scan_properties[scan][image]['x0'] = float(curLine[1])
                        scan_properties[scan][image]['y0'] = float(curLine[2])
                    elif curLine[0] == "(0028.0030)":
                        scan_properties[scan][image]['dx'] = float(curLine[1])
                        scan_properties[scan][image]['dy'] = float(curLine[2])

        # Update masks from contours
        contour_path = scan_path + '\\contours'
        for contour in os.listdir(contour_path):
            file_path = contour_path + '\\' + contour
            image = int(contour.split(".")[0])
            structure = int(contour.split(".")[1])
            region = np.zeros((ORIG_IMAGE, ORIG_IMAGE), dtype=np.uint8)
            # Check if tumor region, if not then continue
            if structure != structure_variables[scan]:
                continue
            x0 = scan_properties[scan][image]['x0']
            y0 = scan_properties[scan][image]['y0']
            dx = scan_properties[scan][image]['dx']
            dy = scan_properties[scan][image]['dy']
            contour_line = []
            with open(file_path, mode='r') as cnt:
                for line in cnt:
                    curLine = line.split(",")
                    for i in range(0, len(curLine), 3):
                        x = (float(curLine[i]) - x0) / dx
                        y = (float(curLine[i + 1]) - y0) / dy
                        region[int(y), int(x)] = 255
                        contour_line.append([int(x), int(y)])
            temp = cv2.fillConvexPoly(region, np.array(contour_line), (255, 255, 255))
            # print(str(image))
            region = temp[crop_low:crop_high, crop_low:crop_high]
            index = scan_properties[scan][image]['index']
            Y[index, 0] = region
            '''
            print(image)
            print(Y[index].shape)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 512, 512)
            cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('mask', 512, 512)
            cv2.imshow('image', X[index, 0])
            cv2.imshow('mask', Y[index, 0])
            cv2.waitKey(0)
            '''
        id1 += id2
        X.flush()
        Y.flush()


def create_test(path, X):
    scan_name = []
    slice_id = []
    auxiliary_properties = {}
    id1 = 0
    for scan in os.listdir(path):
        if scan == '._.DS_Store' or scan == '.DS_Store':
            continue
        scan_path = path + '\\' + scan
        auxiliary_properties[scan] = {}
        auxiliary_path = scan_path + '\\auxiliary'
        for aux in os.listdir(auxiliary_path):
            file_path = auxiliary_path + '\\' + aux
            image = int(aux.split(".")[0])
            auxiliary_properties[scan][image] = {}
            with open(file_path, mode='r') as ax:
                for line in ax:
                    curLine = line.split(",")
                    if curLine[0] == "(0020.0032)":
                        auxiliary_properties[scan][image]['x0'] = float(curLine[1])
                        auxiliary_properties[scan][image]['y0'] = float(curLine[2])
                    elif curLine[0] == "(0028.0030)":
                        auxiliary_properties[scan][image]['dx'] = float(curLine[1])
                        auxiliary_properties[scan][image]['dy'] = float(curLine[2])

        slice_path = scan_path + '\\pngs'
        id2 = 0
        for image in os.listdir(slice_path):
            scan_name.append(scan)
            slice_id.append(int(image.split(".")[0]))
            image_path = slice_path + '\\' + image
            img = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
            img = img[crop_low:crop_high, crop_low:crop_high]
            cv2.normalize(img.copy(), img, 0, 65535, cv2.NORM_MINMAX)
            X[id1 + id2, 0] = np.array(img)
            id2 += 1
        id1 += id2
        X.flush()
    scan_name = np.array(scan_name)
    slice_id = np.array(slice_id)
    np.save('scan_test.npy', scan_name)
    np.save('slice_test_id.npy', slice_id)
    myShelvedDict = shelve.open("test_auxiliary.db")
    myShelvedDict["auxiliary_test_data"] = auxiliary_properties


Xtrain, Ytrain = count_train_images(train_dir, isTrain=True)
Xtest = count_train_images(test_dir, isTrain=False)
create_train(train_dir, Xtrain, Ytrain)
create_test(test_dir, Xtest)
