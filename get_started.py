import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
import pandas as pd


def createOverlay(im,mask,color=(0,1,0),contour=True):
    if len(im.shape)==2:
        im = np.expand_dims(im,axis=-1)
        im = np.repeat(im,3,axis=-1)
    elif len(im.shape)==3:
        if im.shape[-1] != 3:
            ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)

    else:
        ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)
   
    if contour:
        bw = find_boundaries(mask,mode='thick') #inner
    else:
        bw = mask
    for i in range(0,3):
        im_temp = im[:,:,i]
        im_temp = np.multiply(im_temp,np.logical_not(bw)*1)
        im_temp += bw*color[i]
        im[:,:,i] = im_temp
    return im

    
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

#%%
#_____________________________________
# LOAD DATA
#_____________________________________  


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
# to get the nodule texture simply
texture = int(metadata[metadata['Filename']==nodule_names[index]]['texture'])




nb = [0,120] #list of the image index to study

#%%
#_____________________________________
# SHOW IMAGES
#_____________________________________



plot_args={}
plot_args['vmin']=0
plot_args['vmax']=1
plot_args['cmap']='gray'


for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    #since we have volume we must show only a slice
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(getMiddleSlice(nodule),**plot_args)
    ax[1].imshow(getMiddleSlice(mask),**plot_args)
    plt.show()

#if instead you want to overlay
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    over = createOverlay(getMiddleSlice(nodule),getMiddleSlice(mask))
    #since we have volume we must show only a slice
    fig,ax = plt.subplots(1,1)
    ax.imshow(over,**plot_args)
    plt.show()

#%%
#________________________________
# APPLY A MASK TO A NODULE 
#________________________________
    
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    masked = nodule*mask
    #since we have volume we must show only a slice
    fig,ax = plt.subplots(1,1)
    ax.imshow(getMiddleSlice(masked),**plot_args)
    plt.show()

#%%
#________________________________
# OTHER ALGORITHMS TO HELP YOU GET STARTED
#________________________________
    
from skimage.filters.rank import entropy
from skimage.morphology import disk


#mean intensity of a nodule
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    intens = np.mean(nodule[mask!=0])
    print('The intensity of nodule',str(n),'is',intens)
    
#sample points from a nodule mask
np.random.seed(0)
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    
    sampled = np.zeros(mask.shape)
    
    loc = np.nonzero(mask)
    
    indexes = [x for x in range(loc[0].shape[0])]
    np.random.shuffle(indexes)
    
    #get 10% of the points
    indexes = indexes[:int(len(indexes)*0.1)]
    
    sampled[loc[0][indexes],loc[1][indexes],loc[2][indexes]]=True
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(getMiddleSlice(nodule),**plot_args)
    ax[1].imshow(getMiddleSlice(sampled),**plot_args)
    plt.show()    


#create a simple 2 feature vector for 2D segmentation
np.random.seed(0)
features = []
labels = []
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    
    nodule = getMiddleSlice(nodule)
    mask = getMiddleSlice(mask)

    
    #collect itensity and local entropy
    
    entrop = np.ravel(entropy(nodule,disk(5)))
    inten = np.ravel(nodule)
    
    
    labels.append([1 for x in range(int(np.sum(mask)))])
    
    
    features.append([entrop,inten])

    entrop = np.ravel(entropy(nodule==0,disk(5)))
    inten = np.ravel(nodule==0)
    features.append([entrop,inten])
    labels.append([0 for x in range(int(np.sum(mask==0)))])

    
features = np.hstack(features).T
labels = np.hstack(labels)
    
   
#create a simple 2 feature vector for 2D texture analysis
np.random.seed(0)
features = []
labels = []
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    
    nodule = getMiddleSlice(nodule)
    mask = getMiddleSlice(mask)
    
    texture = int(metadata[metadata['Filename']==nodule_names[n]]['texture'])

    
    #collect itensity and local entropy
    
    entrop = np.mean(entropy(nodule,disk(5)))
    inten = np.mean(nodule)
    
    
    labels.append(texture)
    
    
    features.append([entrop,inten])

features_tex = np.vstack(features)
labels_tex = np.hstack(labels) 