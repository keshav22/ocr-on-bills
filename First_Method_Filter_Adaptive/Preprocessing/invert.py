# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:07:25 2019

@author: kesha
"""
from PIL import Image
import PIL.ImageOps
import os
def convertb2wtow2b():
    list1 = os.listdir('./output/')
    for item in list1:
        image = Image.open('./output/'+item)
        img = PIL.ImageOps.invert(image)
        img.save('./input_invert/'+item)
        
convertb2wtow2b()