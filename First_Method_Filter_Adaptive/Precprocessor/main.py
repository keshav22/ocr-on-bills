# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:39:26 2019

@author: kesha
"""

import cv2
import numpy as np

# Read image and search for contours. 
img = cv2.imread('rotatec.jpg')
#cv2.imshow('image',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# Create first mask used for rotation.
mask = np.ones(img.shape, np.uint8)*255

# Draw contours on the mask with size and ratio of borders for threshold.
for cnt in contours:
    size = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    if 10000 > size > 500 and w*2.5 > h:
        cv2.drawContours(mask, [cnt], -1, (0,0,0), -1)

# Connect neighbour contours and select the biggest one (text).
kernel = np.ones((50,50),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
gray_op = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
_, threshold_op = cv2.threshold(gray_op, 150, 255, cv2.THRESH_BINARY_INV)
contours_op, hierarchy_op = cv2.findContours(threshold_op, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = max(contours_op, key=cv2.contourArea)

# Create rotated rectangle to get the angle of rotation and the 4 points of the rectangle.
_, _, angle = rect = cv2.minAreaRect(cnt)
(h,w) = img.shape[:2]
(center) = (w//2,h//2)

# Rotate the image.
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (int(w),int(h)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

# Create bounding box for rotated text (use old points of rotated rectangle).
box = cv2.boxPoints(rect)
a, b, c, d = box = np.int0(box)
bound =[]
bound.append(a)
bound.append(b)
bound.append(c)
bound.append(d)
bound = np.array(bound)
(x1, y1) = (bound[:,0].min(), bound[:,1].min())
(x2, y2) = (bound[:,0].max(), bound[:,1].max())
cv2.drawContours(img,[box],0,(0,0,255),2)

# Crop the image and create new mask for the final image.
rotated = rotated[y1:y2, x1:x2]
mask_final = np.ones(rotated.shape, np.uint8)*255

# Remove noise from the final image.
gray_r = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
_, threshold_r = cv2.threshold(gray_r, 150, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(threshold_r,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    size = cv2.contourArea(cnt)
    if size < 500:
        cv2.drawContours(threshold_r, [cnt], -1, (0,0,0), -1)

# Invert black and white.
final_image = cv2.bitwise_not(threshold_r)

# Display results.

cv2.imshow('final', final_image)
cv2.imshow('rotated', rotated)