# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:20:25 2018

@author: Asus
"""

#%% IMPORTS --------------------------------------------------------------------------------------------------
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter as gaussian
from skimage.morphology import square
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.filters import gabor, gabor_kernel
from skimage import util 
from skimage.filters.rank import entropy
from skimage.morphology import disk
#%% ----------------------------------------------------------------------------------------------------------
#2D NODULE SEGMENTATION
#1-Various gaussian filters 
#2-Compute 2x2 Hessian Matrix
#3-2 eigen-values
#4-Compute Vcomb=int(int(SI,CV),Vmed)
#5-Max Vcomb
#6-Close

#%% FUNCTIONS ------------------------------------------------------------------------------------------------
def findExtension(directory,extension='.npy'):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
            full_path += [os.path.join(directory,file)]
            
    files.sort()
    full_path.sort()
    return files, full_path

def getMiddleSlice(volume):
    sh = volume.shape
    return volume[...,np.int(sh[-1]/2)]

#%% LOAD DATA ---------------------------------------------------------------------------------------------


#find the current working directory
curr_path = os.getcwd()

#find the files
nodule_names, nodules = findExtension(os.path.join(curr_path,'images'))
#remove the extension from the nodule names
nodule_names = [os.path.splitext(x)[0] for x in nodule_names]

mask_names, masks = findExtension(os.path.join(curr_path,'masks'))

#read the metadata
metadata = pd.read_excel('ground_truth.xls')


#to load an images you can simply do
index = 20
example = np.load(nodules[index])
#to get the nodule texture simply
texture = int(metadata[metadata['Filename']==nodule_names[index]]['texture'])

nb = list(range(0, 6)) #list of the image index to study


#%% GAUSSIAN FILTERING AND DATA ORGANIZATION

sigmas=list(np.arange(0.5, 3.5, 0.5))

images=[]
nodules_array=[]
non_nodules_array=[]
for n in range(0,134):
    img=getMiddleSlice(np.load(nodules[n]))
    filtered=gaussian(img,0.5)
    images.append(filtered)
    nodules_array.append(getMiddleSlice(np.load(masks[n])))
    non_nodules_array.append(util.invert(getMiddleSlice(np.load(masks[n]))))
    
train_img, test_img, train_nodules, test_nodules, train_non_nodules, test_non_nodules = train_test_split(images, nodules_array, non_nodules_array, test_size=0.3, random_state=42)
    

#%% GABOR FILTER

#Kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    print(theta)
    for sigma in (3, 5):
        for frequency in (0.05, 0.10):
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


filtered_train=[]
for img in train_img:
    filtered_ims = []   
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (6, 7):
            for frequency in (0.06, 0.07):
                filt_real, filt_imag = gabor(img, frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                filtered_ims.append(filt_real)
    filtered_train.append(filtered_ims)

#%% FEATURES GABOR FILTER + INTENSITY

nr_descriptors = 4*2*2
nr_features=nr_descriptors+1+1
nr_images = 93

sum=0
for nodule in train_nodules:
    sum=sum+np.count_nonzero(nodule == 1)

X = np.zeros([2*sum, nr_features])

y=[]
j=0
for k in range(len(train_img)):
    nr_examples = np.count_nonzero(train_nodules[k] == 1)
    Xi = np.zeros([2*nr_examples, nr_descriptors])
    X_int=np.zeros([2*nr_examples,1])
    img=np.ravel(train_img[k])
    entrop = np.ravel(entropy(train_img[k],disk(5)))
    descriptor = 0
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (6,7):
            for frequency in (0.06, 0.07):
                filt_real, filt_imag = gabor(train_img[k], frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                gabor_descriptor_nodule = filt_real[:,:][train_nodules[k]==True]
                gabor_descriptor_non_nodule = filt_real[:,:][train_non_nodules[k] == True]
                
                # randomly choose which nodule examples to use
                nodule_random_sample_indexes = \
                    np.random.randint(len(gabor_descriptor_nodule), size=nr_examples)
                    
                # randomly choose which non-nodule examples to use
                non_nodule_random_sample_indexes = \
                    np.random.randint(len(gabor_descriptor_non_nodule), size=nr_examples)    
                # Store them in X_gabor
                X_gabor = np.hstack([gabor_descriptor_non_nodule[non_nodule_random_sample_indexes], \
                                     gabor_descriptor_nodule[nodule_random_sample_indexes]])
                # insert them in our global training data matrix            
                Xi[:, descriptor] = X_gabor
                
                # Note that if we were looping on more than one image, 
                # we should create y in a different way. In this case, 
                # we can create it from the beginning.
                descriptor = descriptor+1
                
    X[j:j+2*nr_examples,0:nr_descriptors]=Xi
    X_int=np.hstack([img[nodule_random_sample_indexes], \
                    img[non_nodule_random_sample_indexes]])
    X[j:j+2*nr_examples,nr_descriptors]=X_int
    X_ent=np.hstack([entrop[nodule_random_sample_indexes], \
                entrop[non_nodule_random_sample_indexes]])
    X[j:j+2*nr_examples,nr_descriptors+1]=X_ent
    j=j+2*nr_examples
    y.append(np.zeros([nr_examples]))
    y.append(np.ones([nr_examples]))

Y=y[0]
for i in range(1,len(y)):
    Y=np.hstack((Y,y[i]))
 
#%% SVM
         
#Split in test and validation            
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.33, random_state=42)

#Normalization
X_train_scaled=StandardScaler().fit_transform(X_train)
X_val_scaled=StandardScaler().fit_transform(X_val)

#SVM training
classes=2
clf=svm.SVC(kernel='rbf', C=classes, probability=True).fit(X_train_scaled, y_train)
clf.fit(X_train_scaled, y_train)
s=clf.score(X_val_scaled, y_val)
print(s)

#%% TEST DATA
'''
from skimage import filters
n=6
nodule = test_img[n]
val = filters.threshold_otsu(nodule)
test_mask=nodule>val

nr_descriptors = 4*2*2

nr_images = 1
# Note that now the number of examples is the entire size of the image
nr_pixels = nodule.shape[0]*nodule.shape[1]

nr_examples_per_image = nr_pixels*nr_images

X_test = np.zeros([nr_examples_per_image * nr_images, nr_features])

descriptor = 0
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (6,7):
        for frequency in (0.06, 0.07):
            filt_real, filt_imag = gabor(nodule, frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma)
            gabor_descriptor = filt_real[:,:].ravel() # .ravel() flattens an image
            
            X_test[:, descriptor] = gabor_descriptor
            descriptor = descriptor+1

X_test[:,nr_descriptors]=np.ravel(nodule)
X_test[:,nr_descriptors+1]=np.ravel(entropy(nodule,disk(5)))
X_test_scaled=StandardScaler().fit_transform(X_test)

pred=clf.predict_proba(X_test_scaled)

print(pred.shape, pred[2600,:])

pred_im = pred[:,1].reshape(nodule.shape)

val = filters.threshold_otsu(nodule)
otsu_mask=nodule > val

f, axarr = plt.subplots(1, 3, figsize=(20, 6))
axarr[0].imshow(nodule, cmap='gray')
axarr[1].imshow(pred_im, cmap='gray', interpolation='nearest')
axarr[2].imshow(test_nodules[n], cmap='gray')
plt.show();
'''