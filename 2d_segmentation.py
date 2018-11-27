# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 22:53:26 2018

@author: Asus
"""
#%% IMPORTS --------------------------------------------------------------------------------------------------
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
import pandas as pd
from scipy.ndimage.filters import gaussian_filter as gaussian

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
#index = 20
#example = np.load(nodules[index])
# to get the nodule texture simply
#texture = int(metadata[metadata['Filename']==nodule_names[index]]['texture'])

nb = list(range(0, 6)) #list of the image index to study

#%% GAUSSIAN FILTERS

sigmas=list(np.arange(0.5, 3.5, 0.5))

n=0

img=getMiddleSlice(np.load(nodules[n]))

#for sigma in sigmas:
#    filtered=gaussian(img,sigma)
#    
#    print("sigma=",sigma)
#    #Image filtering results
#    plot_args={}
#    plot_args['vmin']=0
#    plot_args['vmax']=1
#    plot_args['cmap']='gray'
#    fig,ax = plt.subplots(1,2)
#    ax[0].imshow(img,**plot_args)
#    ax[1].imshow(filtered,**plot_args)
#    plt.show()
    
filtered=gaussian(img,1)

#%% HESSIAN MATRIX
# H = [[Gxx, Gxy],[Gxy, Gyy]]
    
gradient1 = np.gradient(filtered)

Gx=gradient1[0]
Gy=gradient1[1]

gradient2 = np.gradient(Gx)
gradient3 = np.gradient(Gy)

Gxx=gradient2[0]
Gxy=gradient2[1]
Gyx=gradient3[0]
Gyy=gradient3[1]

