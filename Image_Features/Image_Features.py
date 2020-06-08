# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:31:21 2020

@author: Abubakar
"""
import matplotlib.pyplot as plt
import numpy as np
import math 
import imageio as im
np.seterr(over='ignore')
class ImageFeature():
  
    def __init__(self, image, size):     
        self.image = image
        self.size = size
        self.no_of_feature = 12
        self.feature_vector = np.zeros(self.no_of_feature)
        

    def compute_feature_vector(self,patch):
        self.feature_vector[0] = self.mean(patch)
        self.feature_vector[1] = self.median(patch)
        self.feature_vector[2] = self.variance(patch)
        self.feature_vector[3] = self.standard_deviation(patch)
        self.feature_vector[4] = self.skewness(patch)
        self.feature_vector[5] = self.kurtosis(patch)
        self.feature_vector[6] = self.mean_abs_deviation(patch)
        self.feature_vector[7] = self.med_abs_deviation(patch)
        self.feature_vector[8] = self.local_contrast(patch)
        self.feature_vector[9] = self.local_prob(patch,195)
        self.feature_vector[10] = self.percentile25(patch)
        self.feature_vector[11] = self.percentile75(patch)
        return  self.feature_vector
    
    def mean(self,patch):
        patch_sum = 0
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                patch_sum = patch_sum + patch[row,col]
        return patch_sum/patch.size
    def median(self,patch):
        i = 0
        patch_list = np.zeros(patch.size)
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                patch_list[i] = patch[row,col]
                i = i + 1
        patch_list = np.sort(patch_list)
        
        return patch_list[(patch.size-1)//2]
    def variance(self,patch):
        patch_sum = 0
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                patch_sum += ((patch[row][col] - np.mean(patch))**2)
        return np.floor(patch_sum/(patch.size-1))
    def standard_deviation(self,patch):
        return math.sqrt(self.variance(patch))
    
    def skewness(self,patch):
        patch_sum = 0
        patch_row,patch_col = patch.shape
        mean = self.mean(patch)
        sd = self.standard_deviation(patch)
        for row in range(patch_row):
            for col in range(patch_col):
                patch_sum = patch_sum + (patch[row,col] - mean)**3
        return patch_sum//((patch.size-1) * np.power(sd,3))
    
    def kurtosis(self,patch):
        patch_sum = 0
        patch_row,patch_col = patch.shape
        mean = self.mean(patch)
        sd = self.standard_deviation(patch)
        for row in range(patch_row):
            for col in range(patch_col):
                patch_sum = patch_sum + (patch[row,col] - mean)**4
        return patch_sum//((patch.size-1) * np.power(sd,4))
        
    def mean_abs_deviation(self,patch):
        patch_sum = 0
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                patch_sum = patch_sum + abs((patch[row,col] - self.mean(patch)))
                
        return patch_sum/patch.size
    
    def med_abs_deviation(self,patch):
        A = []
        for k in range(patch.shape[0]):
            for l in range(patch.shape[1]):
                A.append(patch[k][l])
        A.sort()
        med_index = (patch.size-1)//2
        A1 = []
        for k in range(patch.shape[0]):
            for l in range(patch.shape[1]):    
                y = patch[k][l] - A[med_index]
                A1.append(abs(y))
        A1.sort()
        
        medad_index = (patch.size-1)//2
        return A1[medad_index]
    
    
    def local_contrast(self,patch):
        return np.max(patch) - np.min(patch)
    
    def local_prob(self,patch,k):
        patch_k_count = 0
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                if(patch[row,col] == k):
                    patch_k_count = patch_k_count + 1
        return patch_k_count/patch.size
    
    def percentile25(self,patch):
        i = 0
        patch_list = np.zeros(patch.size)
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                patch_list[i] = patch[row,col]
                i = i + 1
        
        patch_list = np.sort(patch_list)
        return patch_list[math.ceil((patch.size) * 0.25)]
    
    
    def percentile75(self,patch):
        i = 0
        patch_list = np.zeros(patch.size)
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                patch_list[i] = patch[row,col]
                i = i + 1
        patch_list = np.sort(patch_list)
        return patch_list[math.ceil((patch.size) * 0.75)]
    
    
    
    def convolution(self):
        i_row, i_col = self.image.shape
        #k_row, k_col = self.size
        output = np.zeros([i_row,i_col,self.no_of_feature])
        pad_height = (self.size - 1) // 2
        pad_width = (self.size - 1) // 2
     
        padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width)))
     
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = self.image
        
        for row in range(i_row):
            for col in range(i_col):
                output[row, col] = self.compute_feature_vector(padded_image[row:row + self.size, col:col + self.size])
                
        return output



def extract_featured_image(output,shape,feature_index):
    i_row, i_col = shape
    image = np.zeros(shape)
    for row in range(i_row):
        for col in range(i_col):
            image[row,col] = output[row,col,feature_index]
    
    return image
        
img = im.imread("original.png")
features = ImageFeature(img,5)
output = features.convolution()
mean_image = extract_featured_image(output,img.shape,0)
mean_image = mean_image.astype(np.uint8)
median_image = extract_featured_image(output,img.shape,1)
median_image = median_image.astype(np.uint8)
varience_image = extract_featured_image(output,img.shape,2)
varience_image = varience_image.astype(np.uint8)
standard_image = extract_featured_image(output,img.shape,3)
standard_image = standard_image.astype(np.float)
skewnwss_image = extract_featured_image(output,img.shape,4)
skewnwss_image = skewnwss_image.astype(np.float)
kurtosis_image = extract_featured_image(output,img.shape,5)
kurtosis_image = kurtosis_image.astype(np.float)
mean_abs_image = extract_featured_image(output,img.shape,6)
mean_abs_image = mean_abs_image.astype(np.float)
med_abs_image = extract_featured_image(output,img.shape,7)
med_abs_image = med_abs_image.astype(np.uint8)
lcontrast_image = extract_featured_image(output,img.shape,8)
lcontrast_image = lcontrast_image.astype(np.uint8)
lprob_image = extract_featured_image(output,img.shape,9)
lprob_image = lprob_image.astype(np.float)
per25_image = extract_featured_image(output,img.shape,10)
per25_image = per25_image.astype(np.uint8)
per75_image = extract_featured_image(output,img.shape,11)
per75_image = per75_image.astype(np.uint8)
plt.imshow(mean_image,cmap = "gray")
plt.figure()
plt.imsave("mean_image.png",mean_image)
plt.imshow(median_image,cmap = "gray")
plt.figure()
plt.imsave("median_image.png",median_image)
plt.imshow(varience_image,cmap = "gray")
plt.figure()
plt.imsave("varience_image.png",varience_image)
plt.imshow(standard_image,cmap = "gray")
plt.figure()
plt.imsave("standard_image.png",standard_image)
plt.imshow(skewnwss_image,cmap = "gray")
plt.figure()
plt.imsave("skewnwss_image.png",skewnwss_image)
plt.imshow(kurtosis_image,cmap = "gray")
plt.figure()
plt.imsave("kurtosis_image.png",kurtosis_image)
plt.imshow(mean_abs_image,cmap = "gray")
plt.figure()
plt.imsave("mean_abs_image.png",mean_abs_image)
plt.imshow(med_abs_image,cmap = "gray")
plt.figure()
plt.imsave("med_abs_image.png",med_abs_image)
plt.imshow(lcontrast_image,cmap = "gray")
plt.figure()
plt.imsave("lcontrast_image.png",lcontrast_image)
plt.imshow(lprob_image,cmap = "gray")
plt.figure()
plt.imsave("lprob_image.png",lprob_image)
plt.imshow(per25_image,cmap = "gray")
plt.figure()
plt.imsave("per25_image.png",per25_image)
plt.imshow(per75_image,cmap = "gray")
plt.imsave("per75_image.png",per75_image)



###########################Testing for 3 by 3 image
#img = np.zeros([3,3])
#img[0,0] = 1
#img[0,1] = 2
#img[0,2] = 3
#img[1,0] = 4
#img[1,1] = 5
#img[1,2] = 6
#img[2,0] = 7
#img[2,1] = 8
#img[2,2] = 9
#print(output[0,0,0])
#output[0,0] = 1
#output[0,1] = 2
#output[0,2] = 3
#output[1,0] = 4
#output[1,1] = 5
#output[1,2] = 6
#output[2,0] = 7
#output[2,1] = 8
#output[2,2] = 9
#i = 0
#patch_list = np.zeros(output.size)
#patch_row,patch_col = output.shape
#for row in range(patch_row):
#    for col in range(patch_col):
#        patch_list[i] = output[row,col]
#        i = i + 1
#patch_list = np.sort(patch_list)
#print(patch_list)
#print(patch_list[(output.size-1)//2])
#print(np.sum(output)/output.size)
#print(np.median(output))
#print(3**2)