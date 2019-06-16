# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:47:15 2019

@author: kesha
"""

import numpy as np
import cv2
import os

list1 = os.listdir('./output_dpi/')
#print(list1);
for item in list1:    
    img = cv2.imread('./output_dpi/'+item,0)
# create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv2.imwrite('./input_invert/'+item,cl1)
