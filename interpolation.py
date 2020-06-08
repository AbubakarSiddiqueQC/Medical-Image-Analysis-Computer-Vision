# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:05:50 2020

@author: Abubakar
"""

import cv2
import imageio as im
import matplotlib.pyplot as plt
import numpy as np

img = im.imread('strawberries.jpg')
plt.title("Original_Image")
plt.imshow(img, cmap="gray");
plt.show()
plt.close()
plt.title("Interpolation_Nearest_Image")
near_img = cv2.resize(img,None, fx = 0.1, fy = 0.1, interpolation = cv2.INTER_NEAREST)
plt.imshow(near_img);
plt.show()
plt.close()
plt.title("Interpolation_BiLinear_Image")
bilinear_img = cv2.resize(img,None, fx = 0.1, fy = 0.1, interpolation = cv2.INTER_LINEAR)
plt.imshow(bilinear_img);
plt.show()
plt.close()
plt.title("Interpolation_Cubic_Image")
bicubic_img = cv2.resize(img,None, fx = 0.15, fy = 0.15, interpolation = cv2.INTER_CUBIC)
plt.imshow(bicubic_img);
plt.show()
plt.close()
