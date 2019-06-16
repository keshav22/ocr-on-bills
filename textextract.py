# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:44:23 2019

@author: kesha
"""
import pytesseract
import os
import cv2

def textfile(in_pt,ot_pt):
    list1 = os.listdir(in_pt)
    for item in list1:
        img = cv2.imread(in_pt+item)
        result = pytesseract.image_to_string(img, lang="eng")
        f = open(os.path.join(ot_pt+item +".txt"), 'w')
        f.write(result)
        f.close()
#textfile('./output_dpi/')
