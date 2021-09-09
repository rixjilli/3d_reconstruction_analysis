#!/usr/bin/env python
# coding: utf-8

# # 3D Volume Analysis Functions

# ## Introduction
# These functions were developed at **Boston University** to aid in the analysis of pre-segmented 3D reconsructed volumes. This code was originally written to segment three-phase material, but can be modified to analyze two-phase materials (with the exception of the TPB function, which by definition requires three phases). The TPB density function relies on a wonderful package called "Skan", which is cited below. 

# ### Citations:
# J. H. Lee, H. Moon, H. W. Lee, J. Kim, J. D. Kim, and K. H. Yoon, Solid State Ionics, 148(1), 15 (2002).
# 
# J. Nunez-Iglesias, A. J. Blanch, O. Looker, M. W. Dixon, and L. Tilley, PeerJ, 6(4312), 2018. doi:10.7717/peerj.4312.
# 
# M. Kishimoto, H. Iwai, M. Saito, and H. Yoshida, J. Power Sources, 196(10), 4555 (2011).

# In[2]:


import numpy
from PIL import Image
from scipy import ndimage, misc
import scipy
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import math
import skimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import random
import skan


# ### Rescaling
# This function takes a 3D labeled volume that may have asymmetric voxel sizes, and resamples the volume to produce a labeled volume with the smallest possible symmetric voxel size. The function takes a labeled volume (arr), and the physical dimensions along the x, y, and z directions in arbitrary units (d1, d2, and d3 respectively).

# In[9]:


def rescale(arr,d1,d2,d3):
    sizeArr=numpy.shape(arr)
    vs=[d1/sizeArr[0],d2/sizeArr[1],d3/sizeArr[2]]
    v=max(vs)
    zoomV=((d1/v)/sizeArr[0],(d2/v)/sizeArr[1],1)
    Labeled_us = ndimage.zoom(Labeled, zoomV, mode='nearest')
    print("Voxel size: {:.3f} um".format(v))
    return(Labeled_us)


# ### Average Particle Size 
# This function implements a version of the intercept method that both determines an average particle size and the average particle size when measured in the x, y, and z directions. The function takes a labeled volume (with the phase of interest occuping voxels labeled val) and calculates, in each direction and overall, the average intercept size through the particles in that phase. The "scale" input value should be in units/voxel side length (for example, um/voxel side length). The function returns an array with the x-direction, y-direction, z-direction, and average particle sizes for that phase, and automatically multiplies each with a stereographic coefficient, assuming that each particle is spherical.
# 
# See Lee et. al. for further details about the intercept method.

# In[10]:


def bisector_size(arr,val,scale):
    sizeArr=numpy.shape(arr)
    x_sz=[]
    y_sz=[]
    z_sz=[]
    x_int=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    y_int=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    z_int=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    
    #find x sizes
    ct=0
    for z in range(sizeArr[2]):
        for y in range(sizeArr[1]):
            ct=0
            for x in range(sizeArr[0]):
                if arr[x,y,z]==val:
                    ct+=1
                else:
                    ct=0
                x_int[x,y,z]=ct
    for z in range(sizeArr[2]):
        for y in range(sizeArr[1]):
            for x in range(sizeArr[0]-1):
                if x_int[x,y,z]>0 and x_int[x+1,y,z]==0:
                    x_sz.append(x_int[x,y,z])
            if x_int[-1,y,z]>0:
                x_sz.append(x_int[-1,y,z])
                    
    x_size=numpy.mean(x_sz)*scale
    
    
    #find y sizes
    ct=0
    for z in range(sizeArr[2]):
        for x in range(sizeArr[0]):
            ct=0
            for y in range(sizeArr[1]):
                if arr[x,y,z]==val:
                    ct+=1
                else:
                    ct=0
                y_int[x,y,z]=ct
    for z in range(sizeArr[2]):
        for x in range(sizeArr[0]):
            for y in range(sizeArr[1]-1):
                if y_int[x,y,z]>0 and y_int[x,y+1,z]==0:
                    y_sz.append(y_int[x,y,z])
            if y_int[x,-1,z]>0:
                y_sz.append(y_int[x,-1,z])
                    
    y_size=numpy.mean(y_sz)*scale
                
        
    #find z sizes
    ct=0
    for y in range(sizeArr[1]):
        for x in range(sizeArr[0]):
            ct=0
            for z in range(sizeArr[2]):
                if arr[x,y,z]==val:
                    ct+=1
                else:
                    ct=0
                z_int[x,y,z]=ct
    for y in range(sizeArr[1]):
        for x in range(sizeArr[0]):
            for z in range(sizeArr[2]-1):
                if z_int[x,y,z]>0 and z_int[x,y,z+1]==0:
                    z_sz.append(z_int[x,y,z])
            if z_int[x,y,-1]>0:
                z_sz.append(z_int[x,y,-1])
                    
    z_size=numpy.mean(z_sz)*scale
    
    cat_sz=x_sz+y_sz+z_sz
    cat_size=numpy.mean(cat_sz)*scale
                
    return([1.5*x_size,1.5*y_size,1.5*z_size,1.5*cat_size])


# ### Percolation Fraction
# The "perc" function takes a labeled volume and a phase label, and calculates the fraction of pixels that belong to the largest continuous connected volume of that phase within the sampled volume. This function returns the percolated fraction of that phase, as well as an array containing the coordinates of the voxels in the percolated regions of that phase.

# In[11]:


def perc(arr,phase_label):
    sizeArr=numpy.shape(arr)
    barray=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    for x in range(sizeArr[0]):
        for y in range(sizeArr[1]):
            for z in range(sizeArr[2]):
                if arr[x,y,z]==phase_label:
                    barray[x,y,z]=1
    regions=label(barray, connectivity=3)
    reg_props=regionprops(regions)
    reg_areas=numpy.zeros(len(reg_props))
    for i in range(len(reg_props)):
        reg_areas[i]=reg_props[i].area
    perc_index=(numpy.where(reg_areas == numpy.amax(reg_areas)))[0]
    perc_area=reg_props[int(perc_index)].area
    perc_frac=perc_area/sum(sum(sum(barray)))
    perc_locs=reg_props[int(perc_index)].coords
    return(perc_frac,perc_locs)


# ### Tortuosity
# The tortuosity function calculates the tortuosity of a phase using the random walker method. The function takes a labeled array, the label of the phase of interest (phase_label), the number of walkers you'd like to deploy (walkers), and the number of steps those walkers should take (steps). The function returns the tortuosity values as an array (in the x, y, z directions and on average) as well as the average distance of each walker from its origin (rp, rpx, rpy, rpz) and an array containing the walker time steps. To plot, for example, the distance as a function of time step, plot rp vs. steps. For further details about this calculation, see Kishimoto et. al. cited above.

# In[6]:


def tortuosity(arr,phase_label,walkers,steps):
    sizeArr=numpy.shape(arr)
    
    ## find the tortuosity in the percolated phase
    perc_results=perc(arr,phase_label)
    perc_coords=perc_results[1]
    
    perc_vol=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    for x in range(len(perc_coords)):
        perc_vol[(perc_coords[x,:])[0],(perc_coords[x,:])[1],(perc_coords[x,:])[2]]=1
    
    rp=numpy.zeros(steps)
    rpx=numpy.zeros(steps)
    rpy=numpy.zeros(steps)
    rpz=numpy.zeros(steps)
    for n in range(walkers):
        walk_seed_index=random.randint(0,len(perc_coords))
        walk_seed=(perc_results[1])[walk_seed_index]
        pos=walk_seed
        for s in range(steps):
            xchg=random.randint(-1,1)
            ychg=random.randint(-1,1)
            zchg=random.randint(-1,1)
            postemp=numpy.array([pos[0]+xchg,pos[1]+ychg,pos[2]+zchg])
            if (postemp[0] in range(sizeArr[0])) and (postemp[1] in range(sizeArr[1])) and (postemp[2]in range(sizeArr[2])) and (perc_vol[postemp[0],postemp[1],postemp[2]]==1):
                pos=postemp
            else:
                pos=pos
            rp[s]+=((pos[0]-walk_seed[0])**2+(pos[1]-walk_seed[1])**2+(pos[2]-walk_seed[2])**2)
            rpx[s]+=abs(pos[0]-walk_seed[0])**2
            rpy[s]+=abs(pos[1]-walk_seed[1])**2
            rpz[s]+=abs(pos[2]-walk_seed[2])**2
    
    # get displacement as a function of steps for all space, x, y and z
    rp=rp/walkers
    rpx=rpx/walkers
    rpy=rpy/walkers
    rpz=rpz/walkers
    
    ## Free space tortuosity
    rpfs=numpy.zeros(steps)
    rpxfs=numpy.zeros(steps)
    rpyfs=numpy.zeros(steps)
    rpzfs=numpy.zeros(steps)
    for n in range(walkers):
        walk_seed=numpy.array([random.randint(0,sizeArr[0]),random.randint(0,sizeArr[1]),random.randint(0,sizeArr[2])])
        pos=walk_seed
        for s in range(steps):
            xchg=random.randint(-1,1)
            ychg=random.randint(-1,1)
            zchg=random.randint(-1,1)
            postemp=numpy.array([pos[0]+xchg,pos[1]+ychg,pos[2]+zchg])
            if (postemp[0] in range(sizeArr[0])) and (postemp[1] in range(sizeArr[1])) and (postemp[2]in range(sizeArr[2])):
                pos=postemp
            else:
                pos=pos
            rpfs[s]+=((pos[0]-walk_seed[0])**2+(pos[1]-walk_seed[1])**2+(pos[2]-walk_seed[2])**2)
            rpxfs[s]+=abs(pos[0]-walk_seed[0])**2
            rpyfs[s]+=abs(pos[1]-walk_seed[1])**2
            rpzfs[s]+=abs(pos[2]-walk_seed[2])**2
            
    rpfs=rpfs/walkers
    rpxfs=rpxfs/walkers
    rpyfs=rpyfs/walkers
    rpzfs=rpzfs/walkers
    
    step_array=numpy.arange(1,steps+1)
    
    crp=(numpy.polyfit(step_array,rp,1))[0]
    crpx=(numpy.polyfit(step_array,rpx,1))[0]
    crpy=(numpy.polyfit(step_array,rpy,1))[0]
    crpz=(numpy.polyfit(step_array,rpz,1))[0]
    crpfs=(numpy.polyfit(step_array,rpfs,1))[0]
    crpxfs=(numpy.polyfit(step_array,rpxfs,1))[0]
    crpyfs=(numpy.polyfit(step_array,rpyfs,1))[0]
    crpzfs=(numpy.polyfit(step_array,rpzfs,1))[0]
    
    V_frac=perc_results[0]
    
    tort=(1/V_frac)*(crpfs/crp)
    tortx=(1/V_frac)*(crpxfs/crpx)
    torty=(1/V_frac)*(crpyfs/crpy)
    tortz=(1/V_frac)*(crpzfs/crpz)
    tortvals=[numpy.sqrt(tortx),numpy.sqrt(torty),numpy.sqrt(tortz),numpy.sqrt(tort)]
    
    return(tortvals,rp,rpx,rpy,rpz,step_array)


# ### TPB Density
# TPB density is calculated by the volume expansion method, and takes a labeled volume, the values of each of the three phases of interest as an array, the voxel side length (voxel_size), and the number of dilations you'd like to perform on each of the phases to extract the TPBs. Dilations should be initially set to 1, but can be increased if you are concerned about noise or misclassified pixels in your volume.

# In[12]:


def TPB_per_vol(arr,phase_values,voxel_size,dilations):
    sizeArr=numpy.shape(arr)
    tpb_array=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    for i in phase_values:
        barray=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
        for x in range(sizeArr[0]):
            for y in range(sizeArr[1]):
                for z in range(sizeArr[2]):
                    if arr[x,y,z]==i:
                        barray[x,y,z]=1
        barray_exp=scipy.ndimage.morphology.binary_dilation(barray,iterations=dilations).astype(barray.dtype)
        tpb_array=numpy.add(barray_exp,tpb_array)
    for x in range(sizeArr[0]):
        for y in range(sizeArr[1]):
            for z in range(sizeArr[2]):
                if tpb_array[x,y,z]!=len(phase_values):
                    tpb_array[x,y,z]=0
                else:
                    tpb_array[x,y,z]=1
                    
    perc_array=tpb_array #save array for percolation calculation
    
    #Calculate total TPBs
    tpb_array=skeletonize(tpb_array)
    branch_data = skan.Skeleton(tpb_array, spacing=voxel_size)
    tpb_length=sum(branch_data.path_lengths())
    tpb=tpb_length/((sizeArr[0]*sizeArr[1]*sizeArr[2]*voxel_size**3))
    
    perc_coords=(perc(perc_array,1))[1]
    
    perc_vol=numpy.zeros((sizeArr[0],sizeArr[1],sizeArr[2]), dtype=int)
    for x in range(len(perc_coords)):
        perc_vol[(perc_coords[x,:])[0],(perc_coords[x,:])[1],(perc_coords[x,:])[2]]=1
        
    perc_tpb_array=skeletonize(perc_vol)
    perc_branch_data=skan.Skeleton(perc_tpb_array,spacing=voxel_size)
    perc_tpb_length=sum(perc_branch_data.path_lengths())
    perc_tpb=perc_tpb_length/((sizeArr[0]*sizeArr[1]*sizeArr[2]*voxel_size**3))
    
    return(tpb_array,tpb,perc_tpb)


# In[ ]:




