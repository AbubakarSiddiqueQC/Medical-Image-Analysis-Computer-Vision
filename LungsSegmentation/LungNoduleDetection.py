# -*- coding: utf-8 -*-
"""
Created on Sun May  3 07:36:13 2020

@author: Abubakar
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt 

# load image as grayscale
img = cv2.imread('15.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold input image using otsu thresholding as mask and refine with morphology
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
#kernel = np.ones((11,11), np.uint8)
#kernel1 = np.ones((5,5), np.uint8)
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

#edges = cv2.Canny(mask,0,255)

# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw all contours
mask = cv2.drawContours(mask, contours, -1, (0, 255, 0), 2)
#####to avoid removel of 
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((21,21), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#####
#img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
mask1 = np.ones(gray.shape, np.uint8)
cv2.drawContours(mask1, [biggest_contour], -1, (255, 255, 255), -1)
#img2 = np.ones((img.shape[0],img.shape[1]), np.uint8)



imagem = (255-mask)
#masked_image = cv2.bitwise_and(imagem, gray)
masked_image = cv2.bitwise_and(mask1, imagem)
#masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, kernel)
kernel2 = np.ones((23,23), np.uint8)
masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel2)
final = cv2.bitwise_and(masked_image, gray)
plt.imshow(mask,cmap = "gray")
plt.figure()
plt.imshow(masked_image,cmap = "gray")
plt.figure()
plt.imshow(final,cmap = "gray")
plt.figure()
plt.imshow(img,cmap = "gray")

blur = cv2.GaussianBlur(final,(5,5),0)


#plt.figure()
#plt.imshow(blur,cmap = "gray")
plt.figure()
ret,edge_detected_image = cv2.threshold(blur,127,255,1)#cv2.Canny(blur,100, 300)
plt.imshow(edge_detected_image,cmap = "gray")
kernel = np.ones((5,5), np.uint8)
kernel = np.ones((7,7), np.uint8)
edge_detected_image = cv2.morphologyEx(edge_detected_image, cv2.MORPH_OPEN, kernel)
edge_detected_image = cv2.morphologyEx(edge_detected_image, cv2.MORPH_CLOSE, kernel)

#contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnt_areas = []
#contour_list = []
#for contour in contours:
#    perimeter = cv2.arcLength(contour, True)
#    area = cv2.contourArea(contour)
#    if perimeter == 0:
#        break
#    circularity = 4*np.pi*(area/(perimeter*perimeter))
    #print (circularity)
#    if  circularity > 0.8:
#        contour_list.append(contour)
#raw_image = blur.copy()
#cv2.drawContours(raw_image, contour_list,  -1, (0,255,0), 2)
#plt.figure()
#plt.imshow(raw_image,cmap = "gray")
#ret, inner = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imwrite('retina_masked.png', result)
#plt.figure()
#plt.imshow(inner,cmap = "gray")

# detect circles in the image
output = blur.copy()
circles = cv2.HoughCircles(edge_detected_image,cv2.HOUGH_GRADIENT,1,5,param1=50,param2=10,minRadius=0,maxRadius=45)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
else:
    print("No Nodule find");
	# show the output image
plt.figure()
plt.imshow(output,cmap = "gray")
