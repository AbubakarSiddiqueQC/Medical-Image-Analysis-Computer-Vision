# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:12:07 2020

@author: Abubakar
"""
import numpy as np
import math 
import imageio as im
import matplotlib.pyplot as plt
from skimage.feature.texture import greycomatrix, greycoprops
import sklearn.preprocessing
import cv2
class GLCM():
    def __init__(self, image, size,angle,distance):     
        self.image = image
        self.size = size
        self.angle = angle
        self.distance = distance
        self.no_of_feature = 13
        self.i = 0
    
        self.feature_vector = np.zeros(self.no_of_feature)

    def GLCM0(self,patch,d):
            unique_intensities = np.unique(patch)
            mat = np.zeros([unique_intensities.size,unique_intensities.size],dtype="uint32")
            patch_row,patch_col = patch.shape
            for row in range(patch_row):
                for col in range(patch_col-d):
                    mat_row_index = np.where(unique_intensities == patch[row,col])
                    mat_col_index = np.where(unique_intensities == patch[row,col+d])
                    mat[mat_row_index,mat_col_index] = mat[mat_row_index,mat_col_index] + 1
            return mat
    def GLCM45(self,patch,d):
            unique_intensities = np.unique(patch)
            mat = np.zeros([unique_intensities.size,unique_intensities.size],dtype="uint32")
            patch_row,patch_col = patch.shape
            for row in range(patch_row-d):
                for col in range(patch_col-d):
                    mat_row_index = np.where(unique_intensities == patch[row,col])
                    mat_col_index = np.where(unique_intensities == patch[row+d,col+d])
                    mat[mat_row_index,mat_col_index] = mat[mat_row_index,mat_col_index] + 1
    
            return mat
    def GLCM90(self,patch,d):
            unique_intensities = np.unique(patch)
            mat = np.zeros([unique_intensities.size,unique_intensities.size],dtype="uint32")
            patch_row,patch_col = patch.shape
            for row in range(d,patch_row):
                for col in range(patch_col):
                    mat_row_index = np.where(unique_intensities == patch[row,col])
                    mat_col_index = np.where(unique_intensities == patch[row-d,col])
                    mat[mat_row_index,mat_col_index] = mat[mat_row_index,mat_col_index] + 1
    
            return mat
    def GLCM135(self,patch,d):
            unique_intensities = np.unique(patch)
            mat = np.zeros([unique_intensities.size,unique_intensities.size],dtype="uint32")
            patch_row,patch_col = patch.shape
            for row in range(d,patch_row):
                for col in range(patch_col-d):
                    mat_row_index = np.where(unique_intensities == patch[row,col])
                    mat_col_index = np.where(unique_intensities == patch[row-d,col+d])
                    mat[mat_row_index,mat_col_index] = mat[mat_row_index,mat_col_index] + 1
    
            return mat
    
    
    def GLCM(self,patch,angle,distance,sym = False,norm = False):
        if(angle == 0.0):
            mat = self.GLCM0(patch,distance)
        elif(angle == np.pi/4 or angle == 45.0):
            mat = self.GLCM45(patch,distance)
        elif(angle == np.pi/2 or angle == 90.0):
            mat = self.GLCM90(patch,distance)
        elif(angle == 3*np.pi/4 or angle == 135.0 ):
            mat = self.GLCM135(patch,distance)
        
        i_row, i_col = mat.shape
        GLCMat = np.zeros(mat.shape)
        #symmetric
        if(sym):
            GLCMat = mat + mat.T
        else:
            GLCMat = mat
        
        if(norm):
            sum_ = GLCMat.sum()
            GLCMat = GLCMat.astype(np.float)
            for row in range(i_row):
                for col in range(i_col):
                    if(sum_ == 0):
                        GLCMat[row,col] = 0
                    else:
                        GLCMat[row,col] = GLCMat[row,col]/sum_
        return GLCMat
    
    def ASM(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape[0],patch.shape[1]
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + patch[row,col]**2
            
        return sum_
    
    def contrast(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape[0],patch.shape[1]
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + (patch[row,col] * ((row - col)**2))
                
            
        return sum_
    
    def svar(self,patch):
        sum_ = 0
        mean = self.mean(patch)
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + (patch[row,col] * (row - mean)**2)
            
        return sum_
    
    def dissimilarity(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape[0],patch.shape[1]
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + (patch[row,col] * np.abs(row - col))
            
        return sum_
    
    def homogeneity(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape[0],patch.shape[1]
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + (patch[row,col] / (1+(row - col)**2))
            
        return sum_
    
    def energy(self,patch):
        return np.sqrt(self.ASM(patch))
    
    def entropy(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                if(patch[row,col] != 0):
                    sum_ = sum_ - (patch[row,col] * np.log(patch[row,col]))
        return sum_
        
    def correlation(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape[0],patch.shape[1]
        meanx = self.mean_x(patch)
        meany = self.mean_y(patch)
        sdx = self.standard_x(patch)
        sdy = self.standard_y(patch)
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + ((row * col) * (patch[row,col]) - (meanx - meany))//(sdx*sdy)
        return sum_ 
    
    def clustershade(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape[0],patch.shape[1]
        meanx = self.mean_x(patch)
        meany = self.mean_y(patch)
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + (((row + col - meanx - meany)**3) * patch[row,col])
        return sum_ 
    
    def clusterprom(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape[0],patch.shape[1]
        meanx = self.mean_x(patch)
        meany = self.mean_y(patch)
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + (((row + col - meanx - meany)**4) * patch[row,col])
        return sum_ 
    
    def p_x_plus_y(self,GLCMmat):
    
        row,col = GLCMmat.shape[0],GLCMmat.shape[1]
        k_size = 2 * (GLCMmat.shape[0]-1)
        result = [0] * (k_size +1)
        for i in range(row):
            for j in range(col):
                k = i + j
                result[k] = result[k] + GLCMmat[i,j]
        return result  
    def p_x_minus_y(self,GLCMmat):
        row,col = GLCMmat.shape[0],GLCMmat.shape[1]
        k_size = (GLCMmat.shape[0]-1)
        result = [0] * (k_size +1)
        
        for i in range(row):
            for j in range(col):
                k = abs(i - j)
                result[k] = result[k] + GLCMmat[i,j]
        return result
    
    def sum_avg(self,patch):
        sum_ = 0
        pxplusy = self.p_x_plus_y(patch)
        for i in range(2 * (patch.shape[0]-1)):
            sum_ = sum_ + (i * pxplusy[i])
        return sum_
    
            
            
    def sum_entropy(self,patch):
        sum_ = 0
        pxplusy = self.p_x_plus_y(patch)
        for i in range(2 * (patch.shape[0]-1)):
            if(pxplusy[i] !=0):
                sum_ = sum_ - (pxplusy[i] * np.log(pxplusy[i]))
        return sum_
        
    def dif_entropy(self,patch):
        sum_ = 0
        pxminusy = self.p_x_minus_y(patch)
        for i in range(patch.shape[0]-1):
            if(pxminusy[i] !=0):
                sum_ = sum_ - (pxminusy[i] * np.log(pxminusy[i]))
        return sum_
        
        
    
    
    
    
    
    
    def mean(self,patch):
        sum_ = 0
        patch_row,patch_col = patch.shape
        for row in range(patch_row):
            for col in range(patch_col):
                sum_ = sum_ + (row * patch[row,col])
        return sum_
    
    
        
    def mean_x(self,GLCMmat):
        px = np.sum(GLCMmat, axis=1)
        mean = 0
        for i in range(GLCMmat.shape[0]):
            mean = mean + (i * px[i])
        return mean;
    
    def mean_y(self,GLCMmat):
        py = np.sum(GLCMmat, axis=0)
        mean = 0 
        for j in range(GLCMmat.shape[1]):
            mean = mean + (j * py[j])
        return mean;
    
    def standard_x(self,GLCMmat):
        term_to_sqrt = 0
        px = np.sum(GLCMmat, axis=1)
        meanx = self.mean_x(GLCMmat)
        for i in range(GLCMmat.shape[0]):
            term_to_sqrt = term_to_sqrt + ((px[i]-meanx)**2)
        return math.sqrt(term_to_sqrt/(GLCMmat.shape[0]-1))
    
    def standard_y(self,GLCMmat):
        term_to_sqrt = 0
        py = np.sum(GLCMmat, axis=0)
        meany = self.mean_y(GLCMmat)
        for j in range(GLCMmat.shape[1]):
            term_to_sqrt = term_to_sqrt + ((py[j]-meany)**2)
        return math.sqrt(term_to_sqrt/(GLCMmat.shape[1]-1))
    
    
    
    def convolution(self):
        i_row, i_col = self.image.shape
        #k_row, k_col = self.size
        output = np.zeros([i_row,i_col,self.no_of_feature])
        pad_height = (self.size - 1) // 2
        pad_width = (self.size - 1) // 2
     
        padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width)),dtype = np.uint8)
     
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = self.image
        
        for row in range(i_row):
            for col in range(i_col):
                output[row, col] = self.compute_feature_vector(padded_image[row:row + self.size, col:col + self.size])
                
        return output
    
    def compute_feature_vector(self,patch):
        #GLCMmat = self.GLCM(patch,self.angle,self.distance,sym = True,norm = True)
        #patch = patch.astype(np.uint8)
        range_patch = patch.max() - patch.min()
        shape = patch.shape
        if(range_patch == 0):
                range_patch = range_patch + 1
        patch_scaled = sklearn.preprocessing.minmax_scale(patch.ravel(), feature_range=(0,range_patch)).reshape(shape)
        patch_scaled = patch_scaled.astype('uint8')
        M = greycomatrix(image=patch_scaled, distances=[self.distance], angles=[self.angle], levels=range_patch+1, symmetric=True,normed = True)
        GLCM = np.squeeze(M, axis=2)
        GLCM = np.squeeze(GLCM, axis=2)
        
        
        self.feature_vector[0] = self.ASM(GLCM)
        self.feature_vector[1] = self.contrast(GLCM)
        self.feature_vector[2] = self.dissimilarity(GLCM)
        self.feature_vector[3] = self.homogeneity(GLCM)
        self.feature_vector[4] = self.energy(GLCM)
        self.feature_vector[5] = self.entropy(GLCM)
        self.feature_vector[6] = self.svar(GLCM)
        self.feature_vector[7] = greycoprops(M,prop ='correlation')
        self.feature_vector[8] = self.sum_avg(GLCM)
        self.feature_vector[9] = self.sum_entropy(GLCM)
        self.feature_vector[10] = self.dif_entropy(GLCM)
        self.feature_vector[11] = self.clustershade(GLCM)
        self.feature_vector[12] = self.clusterprom(GLCM)
        
       
        print(self.i)
        self.i = self.i + 1
        return  self.feature_vector


def extract_featured_image(output,shape,feature_index):
    i_row, i_col = shape
    image = np.zeros(shape)
    for row in range(i_row):
        for col in range(i_col):
            image[row,col] = output[row,col,feature_index]
    
    return image

img = im.imread("1.png")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
features = GLCM(grey,5,0,1)
output = features.convolution()
asm = extract_featured_image(output,grey.shape,0)
#asm = asm.astype(np.uint32)
contrast = extract_featured_image(output,grey.shape,1)
contrast = contrast.astype(np.uint8)
dissimilarity = extract_featured_image(output,grey.shape,2)
#dissimilarity = dissimilarity.astype(np.uint8)
homogeneity = extract_featured_image(output,grey.shape,3)
#homogeneity = homogeneity.astype(np.float)
energy = extract_featured_image(output,grey.shape,4)
#energy = energy.astype(np.float)
entropy = extract_featured_image(output,grey.shape,5)
#entropy = entropy.astype(np.float)
svar = extract_featured_image(output,grey.shape,6)
svar = svar.astype(np.uint8)
correlation = extract_featured_image(output,grey.shape,7)
#correlation = correlation.astype(np.uint8)
sum_avg = extract_featured_image(output,grey.shape,8)
#sum_avg = sum_avg.astype(np.uint8)
sum_entropy = extract_featured_image(output,grey.shape,9)
#sum_entropy = sum_entropy.astype(np.uint8)
dif_entropy = extract_featured_image(output,grey.shape,10)
#dif_entropy = dif_entropy.astype(np.uint8)
clustershade = extract_featured_image(output,grey.shape,11)
clustershade = clustershade.astype(np.uint8)
clusterprom = extract_featured_image(output,grey.shape,12)
clusterprom = clusterprom.astype(np.uint8)

plt.imshow(asm,cmap = "gray")
plt.figure()

plt.imshow(contrast,cmap = "gray")
plt.figure()

plt.imshow(dissimilarity,cmap = "gray")
plt.figure()

plt.imshow(homogeneity,cmap = "gray")
plt.figure()

plt.imshow(energy,cmap = "gray")
plt.figure()

plt.imshow(entropy,cmap = "gray")
plt.figure()

plt.imshow(svar,cmap = "gray")
plt.figure()

plt.imshow(correlation,cmap = "gray")
plt.figure()

plt.imshow(sum_avg,cmap = "gray")
plt.figure()

plt.imshow(sum_entropy,cmap = "gray")
plt.figure()

plt.imshow(dif_entropy,cmap = "gray")
plt.figure()

plt.imshow(clustershade,cmap = "gray")
plt.figure()

plt.imshow(clusterprom,cmap = "gray")
plt.figure()








###########################Testing for 5 by 5 image
#img0 = im.imread("original.png")
#img1 = img0[0:5,0:5]
#img = np.array([[1,2,3,2,4],[5,3,2,1,1],[3,2,1,2,1],[1,1,5,2,1],[2,5,4,1,3]])

#img = np.array([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]])
#GLCMmat = GLCM(img,135.0,1,sym = True,norm = True)
#print(correlation(GLCMmat))
#print(standard_x(GLCMmat))
#print(standard_y(GLCMmat))
#print(p_for_currunt_col(GLCMmat,0))

#result = greycomatrix(img, [1], [3*np.pi/4], levels=4,symmetric = True,normed = True)
#print(greycoprops(result,prop ='correlation'))
