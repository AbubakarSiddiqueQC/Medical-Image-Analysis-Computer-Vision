# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:43:09 2020

@author: Abubakar
"""
import imageio as im
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
img = im.imread("original.png")
#img = im.imread("Mammo.jpg")
#plt.imshow(img,cmap = "gray")
r,c = img.shape
a = np.amin(img) 
b = np.amax(img) 
R = b-a
Mp = 255
newimg = np.zeros((r,c),dtype=img.dtype)
for x in range(r):
    for y in range(c):
            newimg[x,y] = round(((img[x,y]-a)/R)*Mp);
th,ret1 = cv.threshold(newimg,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#plt.imshow(ret1,cmap = "gray")
plt.hist(newimg.ravel(),256,[0,255])
plt.show

