#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:01:41 2019

@author: sid
"""

#=======================================================================#
# extract_data.py                                                       #
#=======================================================================#
# usage: extract_data.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR]
#
# This program extracts provision numbers from a set of documents.
#
# optional arguments:
#  -h, --help            show this help message and exit
#  -i INPUT_DIR, --input_dir INPUT_DIR
#                        Input directory for the files to be modified
#  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
#                        Output directory for the files to be modified
#=======================================================================#

#=======================================================================#
# Sample usage:                                                         #
#=======================================================================#
#   python extract_data.py --input_dir ocr/data/ --output_dir ocr/results/
#=======================================================================#
from PIL import Image
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import imutils
from parserplot.parserplot import parser
from parserplot.parserplot import plot
import numpy as np
import os
import cv2
import glob
import shutil
import pytesseract
import re
import time
import argparse
from statistics import mode

regex = r"P\d{17}"
found = {}
results = {}
queue = []
done = []
missing = []
pnr_area = [50, 450, 500, 2000]  # [start_x, start_y, end_x, end_y]


# =============================================================================== #
#    To-do list                                                                   #
# =============================================================================== #
# 0. Provision Number                                                             #
# =============================================================================== #


# =============================================================================== #
#    Threshold Methods                                                            #
# =============================================================================== #
# 1. Binary-Otsu w/ Gaussian Blur (kernel size = 9)                               #
# 2. Binary-Otsu w/ Gaussian Blur (kernel size = 7)                               #
# 3. Binary-Otsu w/ Gaussian Blur (kernel size = 5)                               #
# 4. Binary-Otsu w/ Median Blur (kernel size = 5)                                 #
# 5. Binary-Otsu w/ Median Blur (kernel size = 3)                                 #
# 6. Adaptive Gaussian Threshold (31,2) w/ Gaussian Blur (kernel size = 5)        #
# 7. Adaptive Gaussian Threshold (31,2) w/ Median Blur (kernel size = 5)          #
# =============================================================================== #
print("sid")


def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(argument, "Invalid method")


def crop_image(img, start_x, start_y, end_x, end_y):
    #cropped = img[start_y:end_y, start_x:end_x]
    # return cropped
    orig = img
    edged = cv2.Canny(orig, 100, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    ratio = orig.shape[0] / 500.0

    warped = orig
    # loop over the contours
    for c in cnts:
            # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            return four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

            break

    return warped


def get_string(img_path, method):
    # Read image using opencv
    img = cv2.imread(img_path)
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Crop the areas where provision number is more likely present
    #img = crop_image(img, pnr_area[0], pnr_area[1], pnr_area[2], pnr_area[3])
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    #  Apply threshold to get image with only black and white
    img = apply_threshold(img, method)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    #img= cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    save_path = os.path.join(output_path, file_name +
                             "_filter_" + str(method) + ".jpg")
    cv2.imwrite(save_path, img)

    #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")

    return result


def find_match(regex, text):
    matches = re.finditer(regex, text, re.MULTILINE)
    target = ""
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1

        print("  Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                              end=match.end(), match=match.group()))
        target = match.group()

    return target


def pretty_print(result_dict):
    s = ''
    for key in result_dict:
        s += '# ' + key + ': ' + result_dict[key] + '\n'
    return s


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description="This program extracts provision numbers from a set of documents.")
    #parser.add_argument("-i", "--input_dir", help="Input directory for the files to be modified")
    #parser.add_argument("-o", "--output_dir", help="Output directory for the files to be modified")
    #args = parser.parse_args()

    #input_dir = args.input_dir
    #output_dir = args.output_dir
    input_dir = "./Preprocessing/output/"
    output_dir = "./output/"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    im_names = glob.glob(os.path.join(input_dir, '*.png')) + \
        glob.glob(os.path.join(input_dir, '*.jpg')) + \
        glob.glob(os.path.join(input_dir, '*.jpeg'))
    print(im_names)    
    overall_start_t = time.time()
    for im_name in sorted(im_names):
        queue.append(im_name)

    #print("The following files will be processed and their provision numbers will be extracted: {}\n".format(queue))

    for im_name in im_names:
        start_time = time.time()
        #print("*** The documents that are in the queue *** \n{}\n".format(queue))
        print(im_name)
        print('=======================================================')
        print(('# currently model is being applied on {:s}'.format(im_name)))

        queue.remove(im_name)
        
        file_name = im_name.split(".")[1].split("/")[-1]
        # code for all filterss
        #i = 1
        # while i < 8:

        i = 6
        print("> The filter method " + str(i) + " is now being applied.")
        result = get_string(im_name, i)
        match = find_match(regex, result)
        if match:
            if file_name in found:
                found[file_name].append(match)
            else:
                list = []
                list.append(match)
                found[file_name] = list
        
        f = open(os.path.join(output_dir, file_name+"_filter_" + str(i) + ".txt"), 'w')
        f.write(result)
        f.close()
        i += 1
# ------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------

        pnr = ''
        if file_name in found:
            pnr = mode(found[file_name])
            results[file_name] = pnr
            done.append(file_name)
        else:
            missing.append(file_name)
        end_time = time.time()

        print(
            '# It took ' + str(end_time-start_time) + ' seconds.     \n')

    overall_end_t = time.time()

    print('#=======================================================\n'
          '# Summary \n'
          '# The documents that are successfully processed are: \n' + pretty_print(results) +
          '#=======================================================\n'
          '# The program failed to extract information from: \n'
          '# ' + str(missing) + '\n'
          '#=======================================================\n'
          '# It took ' + str(overall_end_t-overall_start_t) + ' seconds.\n'
          '#=======================================================\n')

parser(os.path.join(output_dir))

plot()
