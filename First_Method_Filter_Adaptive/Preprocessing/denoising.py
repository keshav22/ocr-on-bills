#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:50:35 2019

@author: sid
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import maxflow 
import os
# Important parameter
# Higher values means making the image smoother

smoothing = 110

# Load the image and convert it to grayscale image 
def denoise(path):
    list1 = os.listdir(path)

    for item in list1:
        img = cv2.imread(path+item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = 255 * (img > 128).astype(np.uint8)
# Create the graph.
        g = maxflow.Graph[int]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
        nodeids = g.add_grid_nodes(img.shape)
# Add non-terminal edges with the same capacity.
        g.add_grid_edges(nodeids, smoothing)
# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
        g.add_grid_tedges(nodeids, img, 255-img)

# Find the maximum flow.
        g.maxflow()
# Get the segments of the nodes in the grid.
        sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
        img_denoised = np.logical_not(sgm).astype(np.uint8) * 255
        cv2.imwrite('./output_denoise/'+item, img_denoised)
# Show the result.
#        plt.subplot(121)
#        plt.imshow(img, cmap='gray')
#        plt.title('Binary image')
#        plt.subplot(122)
#        plt.title('Denoised binary image')
#        plt.imshow(img_denoised, cmap='gray')
#        plt.show()

# Save denoised image
#        cv2.imwrite('binary.png', img_denoised)
        

denoise('./output_otsu/')