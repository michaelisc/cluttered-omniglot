import os
import pickle
import hashlib

import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 12.0)
from PIL import Image

from joblib import Parallel, delayed


### Create config file

class DatasetGeneratorConfig():
    
    # Scene image shape
    IMAGE_WIDTH = 96
    IMAGE_HEIGHT = 96
    
    # Target image shape
    TARGET_WIDTH = 32
    TARGET_HEIGHT = 32
    
    # Number of distractors = characters placed behind the target
    DISTRACTORS = 31
    
    # Number of occluders = characters placed atop of the target
    OCCLUDERS = 0
    
    # Percentage of empty images [0,1]
    EMPTY = 0
    
    # Drawer split
    DRAWER_SPLIT = 'all' #one of: 'all', 'train', 'val'
    DRAWER_SPLIT_POINT = 10
    
    # Data augmentation settings
    MAX_ROTATION = 20
    MAX_SHEAR = 10
    MAX_SCALE = 2
    
    # Number of images per parallel job 
    JOBLENGTH = 2000
    
    def set_drawer_split(self):
        
            #split char instances
        if self.DRAWER_SPLIT == 'train':
            self.LOW_INSTANCE = 0
            self.HIGH_INSTANCE = self.DRAWER_SPLIT_POINT
            
        elif self.DRAWER_SPLIT == 'val':
            self.LOW_INSTANCE = self.DRAWER_SPLIT_POINT
            self.HIGH_INSTANCE = 20
            
        elif self.DRAWER_SPLIT == 'all':
            self.LOW_INSTANCE = 0
            self.HIGH_INSTANCE = 20
            
        else:
            print("A drawer split has to be chosen from ['all', 'train', 'val']")


            
            
### Define Data Augmentation Functions

# Define rotation functions
def rot_x(phi,theta,ptx,pty):
    return np.cos(phi+theta)*ptx + np.sin(phi-theta)*pty

def rot_y(phi,theta,ptx,pty):
    return -np.sin(phi+theta)*ptx + np.cos(phi-theta)*pty

# Apply affine transformations and scale characters for data augmentation
def prepare_char(some_char, angle=20, shear=10, scale=2):
    phi = np.radians(np.random.uniform(-angle,angle))
    theta = np.radians(np.random.uniform(-shear,shear))
    a = scale**np.random.uniform(-1,1)
    b = scale**np.random.uniform(-1,1)
    (x,y) = some_char.size
    x = a*x
    y = b*y
    xextremes = [rot_x(phi,theta,0,0),rot_x(phi,theta,0,y),rot_x(phi,theta,x,0),rot_x(phi,theta,x,y)]
    yextremes = [rot_y(phi,theta,0,0),rot_y(phi,theta,0,y),rot_y(phi,theta,x,0),rot_y(phi,theta,x,y)]
    mnx = min(xextremes)
    mxx = max(xextremes)
    mny = min(yextremes)
    mxy = max(yextremes)

    aff_bas = np.array([[a*np.cos(phi+theta), b*np.sin(phi-theta), -mnx],[-a*np.sin(phi+theta), b*np.cos(phi-theta), -mny],[0, 0, 1]])
    aff_prm = np.linalg.inv(aff_bas)
    some_char = some_char.transform((int(mxx-mnx),int(mxy-mny)), 
                                  method = Image.AFFINE, 
                                  data = np.ndarray.flatten(aff_prm[0:2,:]))
    some_char = some_char.resize((int(32*(mxx-mnx)/105),int(32*(mxy-mny)/105)))
    
    return some_char

# Crop scaled images to character size
def crop_image(image):
    im_arr = np.asarray(image)
    lines_y = np.all(im_arr == 0, axis=1)
    lines_x = np.all(im_arr == 0, axis=0)
    k = 0
    l = len(lines_y)-1
    m = 0
    n = len(lines_x)-1
    while lines_y[k] == True:
        k = k+1

    while lines_y[l] == True:
        l = l-1

    while lines_x[m] == True:
        m = m+1

    while lines_x[n] == True:
        n = n-1
    
    cropped_image = image.crop((m,k,n,l))
    #plt.imshow(image.crop((m,k,n,l)))
    return cropped_image

# Color characters with a random RGB color
def color_char(tmp_im):
    size = tmp_im.size
    tmp_im = tmp_im.convert('RGBA')
    tmp_arr = np.asarray(tmp_im)
    rnd = np.random.rand(3)
    stuff = tmp_arr[:,:,0] > 0
    tmp_arr = tmp_arr*[rnd[0], rnd[1], rnd[2], 1]
    tmp_arr[:,:,3] = tmp_arr[:,:,3]*stuff
    tmp_arr = tmp_arr.astype('uint8')
    tmp_im = Image.fromarray(tmp_arr)
    
    return tmp_im




### Define Image Generation Functions

# Generate one image with clutter
def make_cluttered_image(chars, char, n_distractors, config, verbose=0):
    '''Inputs:
    chars: Dataset of characters
    char: target character
    nclutt: number of distractors
    empty: if True do not include target character'''
    
    # While loop added for error handling
    l=0
    while l < 1:
        #initialize image and segmentation mask
        im = Image.new('RGBA', (config.IMAGE_WIDTH,config.IMAGE_HEIGHT), (0,0,0,255))
        seg = Image.new('RGBA', (config.IMAGE_WIDTH,config.IMAGE_HEIGHT), (0,0,0,255))
        
        #generate background clutter
        for j in range(0,n_distractors):
            # draw random character instance
            rnd_char = np.random.randint(0,len(chars))
            rnd_ind = np.random.randint(config.LOW_INSTANCE,config.HIGH_INSTANCE)
            some_char = chars[rnd_char][rnd_ind]
            try:
                # augment random character
                tmp_im = prepare_char(some_char)
                tmp_im = crop_image(tmp_im)
                tmp_im = color_char(tmp_im)
            except:
                if verbose > 0:
                    print('Error generating distractors')
                continue
            # add augmented random character to image
            im.paste(tmp_im, 
                     (np.random.randint(0,im.size[0]-tmp_im.size[0]+1), 
                      np.random.randint(0,im.size[1]-tmp_im.size[1]+1)), 
                     mask = tmp_im)
        
        # if empty: draw another random character instead of the target
        empty = np.random.random() < config.EMPTY
        if empty:
            rnd_char = np.random.randint(0,len(chars))
            rnd_ind = np.random.randint(config.LOW_INSTANCE,config.HIGH_INSTANCE)
            char = chars[rnd_char][rnd_ind]
        
        try:
            # augment target character
            glt_im = prepare_char(char) #transform char
            glt_im = crop_image(glt_im) #crop char
            glt_im_bw = glt_im
            glt_im = color_char(glt_im) #color char
        except:
            if verbose > 0:
                print('Error augmenting target character')
            continue

        # place augmentad target char        
        left = np.random.randint(0,im.size[0]-glt_im.size[0]+1)
        upper = np.random.randint(0,im.size[1]-glt_im.size[1]+1)
        im.paste(glt_im, (left, upper), mask = glt_im)
        
        #make segmentation mask
        if not empty:
            seg.paste(glt_im_bw, (left, upper), mask = glt_im_bw)
        
        
        # generate occlusion
        for j in range(0,config.OCCLUDERS):
            # draw random character
            rnd_char = np.random.randint(0,len(chars))
            rnd_ind = np.random.randint(config.LOW_INSTANCE,config.HIGH_INSTANCE)
            some_char = chars[rnd_char][rnd_ind]
            try:
                # augment occluding character
                tmp_im = prepare_char(some_char)
                tmp_im = crop_image(tmp_im)
                tmp_im = color_char(tmp_im)
            except:
                if verbose > 0:
                    print('Error generating occlusion')
                continue
            # place occluding character
            im.paste(tmp_im, 
                     (np.random.randint(0,im.size[0]-tmp_im.size[0]+1), 
                      np.random.randint(0,im.size[1]-tmp_im.size[1]+1)), 
                     mask = tmp_im)

        
        #convert image from RGBA to RGB for saving    
        im = im.convert('RGB')
        seg = seg.convert('1')
        
        l=l+1
        
    return im, seg

def make_target(chars, char, config, verbose=0):
    '''Inputs:
    chars: Dataset of characters
    char: target character'''
    
    # Legacy while loop to generate multiple targets for data augemntation
    # Multiple targets did not improve performance in our experiments
    l=0
    while l < 1:
        
        try:
            # initialize image
            im = Image.new('RGBA', (config.TARGET_WIDTH,config.TARGET_HEIGHT), (0,0,0,255))

            # augment target character (no scaling is applied)
            glt_im = prepare_char(char, angle=config.MAX_ROTATION, shear=config.MAX_SHEAR, scale=1) #transform char
            glt_im = crop_image(glt_im) #crop char
            glt_im = color_char(glt_im) #color char

            #place target character        
            left = (im.size[0]-glt_im.size[0])//2
            upper = (im.size[1]-glt_im.size[1])//2
            im.paste(glt_im, (left, upper), mask = glt_im)

            #convert image from RGBA to RGB for saving    
            im = im.convert('RGB')

        except:
            if verbose > 0:
                print('Error generating target')
            continue
        
        l=l+1
        
    return im

def make_image(chars, 
               k, 
               config,
               seed=None):
    '''Inputs:
    chars: Dataset of characters
    angle: legacy
    shear: legacy
    scale: legacy
    joblength: number of images to create in each job
    k: job index
    seed: random seed to generate different results in each job
    coloring: legacy'''
    
    # Generate random seed
    np.random.seed(seed)
    
    # Initialize batch data storage
    r_ims = np.zeros((config.JOBLENGTH,config.IMAGE_WIDTH,config.IMAGE_HEIGHT,3), dtype='uint8')
    r_seg = np.zeros((config.JOBLENGTH,config.IMAGE_WIDTH,config.IMAGE_HEIGHT,1), dtype='uint8')
    r_tar = np.zeros((config.JOBLENGTH,config.TARGET_WIDTH,config.TARGET_HEIGHT,3), dtype='uint8')
    
    for i in range(config.JOBLENGTH):
        
        #select a char
        char_char = np.random.randint(0,len(chars))
        char_ind = np.random.randint(config.LOW_INSTANCE,config.HIGH_INSTANCE)
        char = chars[char_char][char_ind]
        
        # choose random number of distractors for datasets with varying clutter
        # selects the one fixed number of distractors in other cases
        n_distractors = np.random.choice(config.DISTRACTORS)
        #generate images and segmentation masks
        ims, seg = make_cluttered_image(chars, char, n_distractors, config)

        #generate targets
        tar = make_target(chars, char, config)
        
        # Append to dataset
        r_ims[i,:,:,:] = ims
        r_seg[i,:,:,0] = seg
        r_tar[i,:,:,:] = tar
    
    return r_ims, r_seg, r_tar



### Multiprocessing Dataset Generation Routine

def generate_dataset(path, 
                     dataset_size, 
                     chars,
                     config,
                     seed=None,
                     save=True, 
                     show=False,
                     checksum=None):
    
    '''Inputs:
    path: Save path
    N: number of images
    chars: Dataset of characters
    char_locs: legacy
    split: train/val split of drawer instances
    save: If True save dataset to path
    show: If true plot generated images'''
    
    t = time.time()
    
    # Define necessary number of jobs
    N = dataset_size
    M = dataset_size//config.JOBLENGTH
    
    # Initialize data
    data_ims = np.zeros((N,config.IMAGE_WIDTH,config.IMAGE_HEIGHT,3), dtype='uint8')
    data_seg = np.zeros((N,config.IMAGE_WIDTH,config.IMAGE_HEIGHT,1), dtype='uint8')
    data_tar = np.zeros((N,config.TARGET_WIDTH,config.TARGET_HEIGHT,3), dtype='uint8')

    # Execute parallel data generation
    #for i in range(0,N):
    #with Parallel(n_jobs=10, verbose=50) as parallel:
    print('Executing %.d tasks'%(M))
    if seed:
        np.random.seed(seed)
        print('Seed fixed')
    seeds = np.unique(np.random.randint(2**32, size=2*M))
    results = Parallel(n_jobs=-1, verbose=50)(delayed(make_image)(chars,
               k, 
               config,
               seed=seeds[k]) for k in range(M))

    # feed results into the dataset
    for i in range(0,M):
        for j in range(config.JOBLENGTH):
            data_ims[i*config.JOBLENGTH+j,:,:,:] = results[i][0][j,...] 
            data_seg[i*config.JOBLENGTH+j,:,:,:] = results[i][1][j,...]
            data_tar[i*config.JOBLENGTH+j,:,:,:] = results[i][2][j,...]
        
            


    #save dataset
    save = save
    if save == True:
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + 'images', data_ims.astype('uint8'))
        np.save(path + 'segmentation', data_seg.astype('uint8'))
        np.save(path + 'targets', data_tar.astype('uint8'))

    #show outputs
    show = show
    if show == True:
        for i in range(0,N):   
            plt.figure
            plt.subplot(131)    
            plt.imshow(data_tar[i,...])
            
            plt.subplot(132)
            plt.imshow(data_ims[i,...])
            
            plt.subplot(133)
            plt.imshow(data_seg[i,...,0])

            plt.show()


    print("Duration:", time.time()-t)
    
    # Test checksum
    last_image = data_ims[-1,...]
    print("Hash:", hashlib.md5(last_image).digest())
    if checksum:
        if hashlib.md5(last_image).digest() == checksum:
            print("Dataset was correctly created!")
        else:
            print("Incorrect hash value!")
            
    
    
### Data loader

def load_dataset(dataset_dir, subset):
    
    assert subset in ['train', 'val-train', 'test-train', 'val-one-shot', 'test-one-shot']
    
    path = os.path.join(dataset_dir, subset)
        
    # Load data in memory mapping mode to reduce RAM usage
    ims = np.load(os.path.join(path, 'images.npy'), mmap_mode='r')
    seg = np.load(os.path.join(path, 'segmentation.npy'), mmap_mode='r')
    tar = np.load(os.path.join(path, 'targets.npy'), mmap_mode='r')
    
    return ims, seg, tar