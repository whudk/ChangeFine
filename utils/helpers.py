import cv2
import numpy as np
import itertools
import operator
import os, csv
# import tensorflow as tf
import math
import time, datetime

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same

    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map

def get_subcls_pred(image, cls):
    if cls < 0:
        cls = abs(cls)
    div_array = np.sum(image,axis=2)
    sub_image = image[:,:,cls]
    x = sub_image / div_array
    return  x*255
def get_subcls_pred_1(image, subcls):
    x = np.zeros((image.shape[0],image.shape[1],len(subcls)),dtype=np.uint8)
    for id,cls in enumerate(subcls):
        if cls == 0:
            x[:,:,0] = reverse_one_hot(image)
        else:
            x[:,:,id] = get_subcls_pred(image,cls)
    return x
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# class_dict = get_class_dict("CamVid/class_dict.csv")
# gt = cv2.imread("CamVid/test_labels/0001TP_007170_L.png",-1)
# gt = reverse_one_hot(one_hot_it(gt, class_dict))
# gt = colour_code_segmentation(gt, class_dict)

# file_name = "gt_test.png"
# cv2.imwrite(file_name,np.uint8(gt))

def computePadSize(src_x_size):
    sub_x = 0
    if src_x_size < 100000:
        if len(str(src_x_size)) == 5:
            sub_x = src_x_size - int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)  # sub-number
            sub_x = np.ceil(sub_x / 1000) * 1000
            sub_x = sub_x + int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)
            sub_x = sub_x - src_x_size

        elif len(str(src_x_size)) == 4:
            sub_x = src_x_size - int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)  # sub-number
            sub_x = np.ceil(sub_x / 100) * 100
            sub_x = sub_x + int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)
            sub_x = sub_x - src_x_size

        elif len(str(src_x_size)) == 3:
            sub_x = src_x_size - int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)  # sub-number
            sub_x = np.ceil(sub_x / 10) * 10
            sub_x = sub_x + int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)
            sub_x = sub_x - src_x_size
        elif len(str(src_x_size)) == 2:
            sub_x = src_x_size - int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)  # sub-number
            sub_x = np.ceil(sub_x / 1) * 1
            sub_x = sub_x + int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)
            sub_x = sub_x - src_x_size
    return int(sub_x)
