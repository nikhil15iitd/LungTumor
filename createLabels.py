import os
import numpy as np
import cv2
import shelve

train_dir = 'C:\\Users\\nikhil\\LungTumor\\example_extracted\\example_extracted'
test_dir = 'C:\\Users\\nikhil\\LungTumor\\provisional_extracted_no_gt\\provisional_extracted_no_gt'
ORIG_IMAGE = 512
RESIZE_IMAGE = 96
scan_properties = {}
# Check the index of tumor structure for each patient (or scan)
structure_variables = {}

tumor_names = ["radiomics_gtv", "radiomics_gtv", "Radiomics_gtv", "radiomics_gtv2", "radiomics_gtv_nw",
               "radiomics_gtvr"]
tumor_names = set(tumor_names)

crop_low = 100
crop_high = 400


def create_train(path):
    X = []
    Y = []
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
            id2 += 1
            image_path = slice_path + '\\' + image
            img = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
            img = img[crop_low:crop_high, crop_low:crop_high]
            cv2.normalize(img.copy(), img, 0, 65535, cv2.NORM_MINMAX)
            img = cv2.resize(img, (RESIZE_IMAGE, RESIZE_IMAGE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img_mask = np.zeros(img.shape, dtype=np.uint8)
            X.append(np.array([img]))
            Y.append(np.array([img_mask]))

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
            # short contours act as noise, avoid them
            if len(contour_line) < 4:
                continue
            temp = cv2.fillConvexPoly(region, np.array(contour_line), (255, 255, 255))
            # print(str(image))
            region = temp[crop_low:crop_high, crop_low:crop_high]
            img_mask = cv2.resize(region.copy(), (RESIZE_IMAGE, RESIZE_IMAGE), interpolation=cv2.INTER_AREA)
            index = scan_properties[scan][image]['index']
            Y[index][0] = img_mask
            '''
            print(image)
            print(Y[index].shape)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', RESIZE_IMAGE, RESIZE_IMAGE)
            cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('mask', RESIZE_IMAGE, RESIZE_IMAGE)
            cv2.imshow('image', X[index][0])
            cv2.imshow('mask', Y[index][0])
            cv2.waitKey(0)
            '''
        id1 += id2

    X = np.array(X)
    Y = np.array(Y)
    np.save('train.npy', X)
    np.save('mask_train.npy', Y)
    del X
    del Y


def create_test(path):
    X = []
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
            id2 += 1
            image_path = slice_path + '\\' + image
            img = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
            img = img[crop_low:crop_high, crop_low:crop_high]
            cv2.normalize(img.copy(), img, 0, 65535, cv2.NORM_MINMAX)
            img = cv2.resize(img, (RESIZE_IMAGE, RESIZE_IMAGE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            X.append(np.array([img]))
        id1 += id2
    X = np.array(X)
    scan_name = np.array(scan_name)
    slice_id = np.array(slice_id)
    np.save('test.npy', X)
    np.save('scan_test.npy', scan_name)
    np.save('slice_test_id.npy', slice_id)
    myShelvedDict = shelve.open("test_auxiliary.db")
    myShelvedDict["auxiliary_test_data"] = auxiliary_properties


create_train(train_dir)
create_test(test_dir)
