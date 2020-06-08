# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:00:32 2020

@author: Abubakar
"""
import imageio as im
import matplotlib.pyplot as plt
import numpy as np
img = im.imread("original.png")
#plt.imshow(img,cmap = "gray")
#plt.axis('off')
#plt.hist(img.ravel(),256,[0,255])
#plt.show
r,c = img.shape
newimg = np.zeros((r,c),dtype=np.uint8)
x=1
y=1
for x in range(r-1):
    for y in range(c-1):
        adaptivethresh = (img[x-1,y-1]+img[x,y-1]+img[x+1,y-1]+img[x+1,y]+img[x-1,y]+img[x-1,y+1]+img[x,y+1]+img[x+1,y+1])/8
        newimg[x,y] =  adaptivethresh
plt.imshow(newimg,cmap = "gray"  )
plt.show
