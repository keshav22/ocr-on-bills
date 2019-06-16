# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:58:03 2019

@author: kesha
"""
import os
from text_area_crop import process_image
from skew import skew
from dpi import dpi
from otsu import otsu
from textextract import textfile
from parserplot.parserplot import parser
from parserplot.parserplot import plot

input_path = './input_org_img/'
output_path = './input_crop/'
skew(input_path, output_path)


input_path = './input_crop/'
output_path = './input_dpi/'
list1 = os.listdir("./input/")
for file in list1:
    try:
        process_image(input_path+file,output_path+file)
    except Exception as e:
        print('%s %s' % ('', e))

input_path = './input_dpi/'
output_path = './input_otsu/'
dpi(input_path,output_path)

input_path = './input_otsu/'
output_path = './output_otsu/'
otsu(input_path,output_path)

input_path = './output_otsu/'
output_path = './finalImages/'
dpi(input_path,output_path)

input_path = './finalImages/'
output_file = './final_text_files/'
textfile(input_path,output_path)

parser('./final_text_files/')
plot()


