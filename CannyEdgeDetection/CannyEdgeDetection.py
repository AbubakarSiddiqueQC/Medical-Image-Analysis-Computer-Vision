# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:51:12 2020

@author: Abubakar
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imageio as im
 
 
def convolution(image, kernel, average=False):
    i_row, i_col = image.shape
    k_row, k_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = (k_row - 1) // 2
    pad_width = (k_col - 1) // 2
 
    padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    
    for row in range(i_row):
        for col in range(i_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + k_row, col:col + k_col])
            if average:
                output[row, col] /= k_row * k_col
    return output
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D
def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=1)
    return convolution(image, kernel, average=True)
def sobel_Operator_Gradient(image):
    hor_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ver_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) 
    new_image_x = convolution(image, hor_filter)
    print("after hor soble")
    plt.figure()
    plt.imshow(new_image_x)
    new_image_y = convolution(image, ver_filter)
    print("after ver soble")
    plt.figure()
    plt.imshow(new_image_y)
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    gradient_theta = np.arctan2(new_image_y, new_image_x)
    gradient_theta = np.rad2deg(gradient_theta)
    gradient_theta += 180
    return (gradient_magnitude,gradient_theta)

def non_max_suppression(gradient_magnitude, gradient_theta):
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_theta[row, col]
            #angle 0
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]
            #angle 45
            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]
            #angle 90
            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]
            #angle 135
            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]
     
            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
    
    return output
        
def threshold(image, low, high, weak):
    output = np.zeros(image.shape)
    strong = 255
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak
    return output

def hysteresis(image, weak):
    image_row, image_col = image.shape
 
    top_to_bottom = image.copy()
 
    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0
 
    bottom_to_top = image.copy()
 
    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0
 
    right_to_left = image.copy()
 
    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0
 
    left_to_right = image.copy()
 
    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0
 
    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
 
    final_image[final_image > 255] = 255
 
    return final_image

img = im.imread("canny.jpg")
if len(img.shape) == 3:
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
else:
    grey_img = img
newimg = np.zeros(grey_img.shape)
#First Step
blurred = gaussian_blur(grey_img, 5)
print("Orignal image")
plt.imshow(grey_img)
print("blurred image")
plt.figure()
plt.imshow(blurred)
#step two
gradient_mag,gradient_theta = sobel_Operator_Gradient(blurred)
print("After Sobel operator image")
plt.figure()
plt.imshow(gradient_mag)
#Step three
non_mexima = non_max_suppression(gradient_mag, gradient_theta)
print("After non maxia image")
plt.figure()
plt.imshow(non_mexima)
#Step four
weak = 50
after_thresh = threshold(non_mexima, 5, 20, weak=weak)
print("After thresholding image")
plt.figure()
plt.imshow(after_thresh)
#Step 5
final = hysteresis(after_thresh, weak)
print("final image")
plt.figure()
plt.imshow(final)
print("final image")

plt.figure()
plt.imshow(cv.Canny(grey_img,100,200))
plt.show
