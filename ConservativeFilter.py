# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:27:33 2020

@author: Abubakar
"""
import imageio as im
import matplotlib.pyplot as plt
import numpy as np
img = im.imread("original.png")
r,c = img.shape
newimg = np.zeros((r,c),dtype=np.uint8)
WindowSize = 3
x=1
y=1
for x in range(r-1):
    for y in range(c-1):
        window = np.array([img[x-1,y-1],img[x,y-1],img[x+1,y-1],img[x+1,y],img[x-1,y],img[x-1,y+1],img[x,y+1],img[x+1,y+1]])
        min1 = np.min(window)
        max1 = np.max(window)
        if(img[x,y] > max1):
            newimg[x,y] = max1
        elif(img[x,y] < min1):
            newimg[x,y] = min1
        else:
            newimg[x,y] = img[x,y]
plt.imshow(newimg,cmap = "gray"  )
plt.show
