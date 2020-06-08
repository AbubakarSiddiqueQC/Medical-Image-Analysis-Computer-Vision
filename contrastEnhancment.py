# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:18:08 2020

@author: Abubakar
"""
import imageio as im
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
img = im.imread("original.png")
#plt.imshow(img,cmap = "gray")
r,c = img.shape
newimg = np.zeros((r,c),dtype=img.dtype)
for x in range(r):
    for y in range(c):
            newimg[x,y] =255-img[x,y]
th,ret1 = cv.threshold(newimg,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(ret1,cmap = "gray")
plt.show
#plt.close()

