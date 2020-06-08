# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:32:27 2020

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
unique, counts = np.unique(img, return_counts=True)
dict(zip(unique, counts))
countsum =np.sum(counts) 
prob = np.zeros(counts.shape,dtype=float)
cdf = np.zeros(counts.shape,dtype=float)
newimg = np.zeros(img.shape,dtype=img.dtype)
for x in range(counts.size):
    prob[x] = counts[x]/countsum
    cdf[x] = np.sum(prob[:x])
cdf = cdf * 255 
for x in range(r):
    for y in range(c):
        for i in range(counts.size):
            if(img[x,y] == unique[i]):
                newimg[x,y] = round(cdf[i])
th,ret1 = cv.threshold(newimg,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(newimg,cmap = "gray")
#plt.hist(newimg.ravel(),256,[0,255])
plt.show