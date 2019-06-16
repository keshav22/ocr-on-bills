#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:59:43 2019

@author: sid
"""
import sys
from PIL import Image
import PIL.ImageOps    
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import os

def skew(path1,path2):
    list1 = os.listdir(path1)
    for item in list1:
        img = im.open(path1+item)

        # convert to binary
        wd, ht = img.size
        pix = np.array(img.convert('1').getdata(), np.uint8)
        bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
        plt.imshow(bin_img, cmap='gray')
        #plt.savefig('binary.png')
    
        delta = 1
        limit = 5
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = find_score(bin_img, angle)
            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]

        data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
        img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
        img = PIL.ImageOps.invert(img)
        img.save(path2+item);

def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

#def converb2wtow2b():
 #   image = Image.open('skew_corrected.png')
 #    img = PIL.ImageOps.invert(img)
  #  inverted_image.save('new_name.png')
#skew('../input/')