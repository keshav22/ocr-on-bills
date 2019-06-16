# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:59:39 2019

@author: kesha
"""

from PIL import Image
import os

def dpi(in_path,ot_path):
    list1 = os.listdir(in_path)
    for item in list1:
        im = Image.open(in_path+item)
        im.save(ot_path+item, dpi=(300,300))