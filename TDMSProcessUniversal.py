# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:24:38 2020

@author: mbro632
"""

import os,sys
file_dir = os.path.dirname('H:/PhD/Python/GitHub/Universal')
sys.path.append(file_dir)

import matplotlib.pyplot as plt
import numpy as np
import time
import DirectorySetupUniversal as directory
import TdmsCodeUniversal as pytdms
import scipy as sci
from scipy import signal,ndimage
import pandas as pd
import glob
import tifffile as tf
import scipy as sci

plt.close('all')

tstart=time.time()
#Data location information  
folderLocation = r'H:/PhD/OCT Data/Compression Test/Calibration/Step 0.05'                                                           
npy = r'\npy_files'                                                             
output = r'\output_files'                                                        
#Git Comment

#PS-OCT Image information
spectra_num = 1024
A_scan_num = 714
B_scan_num = 1
padSize = 2**14 #How much zero padding do you want to do

#Segmentation Values
SurfThresh = 85
depthThresh=823
gaussSigma = 12
scale = (2*padSize + spectra_num)/spectra_num
windowSize = 15


dataLocation = folderLocation
print(dataLocation)
sampleName = os.path.basename(dataLocation)
imageFiles1 = sorted(glob.glob(dataLocation + '/' + 'Ch0_*.tdms'))
imageFiles = sorted(glob.glob(dataLocation + '/' + 'Ch1_*.tdms'))
peakpos=[]
for allImages in imageFiles:
    print(allImages)
    variableName = allImages.split('/')[-1]
    variableName = variableName.split('_',1)[-1]
    variableName = variableName.rsplit('.',1)[0]
    Int = []
    Ret = []
    Ch0Complex = []
    Ch1Complex = []
    t1 = time.time()
    Int,Ret, Ch0Complex,Ch1Complex = directory.loadData(dataLocation,npy,output,variableName,A_scan_num,B_scan_num,padSize)
    t2 = time.time()
    print('It took {:.2f} seconds to load in the data'.format(t2-t1))

    sliced=10
    intdB = 10*np.log10(Int)
    
    plt.figure()
    plt.imshow(intdB,cmap='binary')
    plt.clim([75,110])

    
    plt.figure()
    plt.imshow(Ret,cmap='binary')
    plt.clim([0.2,1.4])



    
    
    # glass= plt.ginput(2)
    # slide = plt.ginput(2)

    # xpointsglass = np.array([i[0] for i in glass]).astype(int)
    # ypointsglass = np.array([i[1] for i in glass]).astype(int)
    
    # glassSurf = []
    # xAxisGlass=np.arange(min(xpointsglass), max(xpointsglass),1)
    # differenceMax = 5
    
    # for i in range(min(xpointsglass), max(xpointsglass)):
    #     surf = np.argmax(meanIntdB[5:max(ypointsglass),i]) + 5
    #     glassSurf = np.append(glassSurf,surf)
        
        
    # pGlass = np.polyfit(xAxisGlass,glassSurf,1)

    # for i in range(0,len(glassSurf)):
    #     lower = int(pGlass[0]*xAxisGlass[i] + pGlass[1] - differenceMax)
    #     upper = int(pGlass[0]*xAxisGlass[i] + pGlass[1] + differenceMax)
    #     if glassSurf[i] not in range(lower,upper):
    #         glassSurf[i] =  pGlass[0]*xAxisGlass[i] + pGlass[1]
        
    # pGlass = np.polyfit(xAxisGlass,glassSurf,1)
        
        
    # plt.plot(xAxisGlass, glassSurf,'rx')
    # plt.plot(xAxisGlass, pGlass[0]*xAxisGlass + pGlass[1],'b')
            
    




    # xpointsslide = np.array([i[0] for i in slide]).astype(int)
    # ypointsslide = np.array([i[1] for i in slide]).astype(int)
    
    # slideSurf = []
    # xAxisSlide=np.arange(min(xpointsslide), max(xpointsslide),1)
    # differenceMax = 5
    
    # for i in range(min(xpointsslide), max(xpointsslide)):
    #     surf = np.argmax(meanIntdB[5:max(ypointsslide),i]) + 5
    #     slideSurf = np.append(slideSurf,surf)
        
        
    # pSlide = np.polyfit(xAxisSlide,slideSurf,1)
    
    # for i in range(0,len(slideSurf)):
    #     lower = int(pSlide[0]*xAxisSlide[i] + pSlide[1] - differenceMax)
    #     upper = int(pSlide[0]*xAxisSlide[i] + pSlide[1] + differenceMax)
    #     if slideSurf[i] not in range(lower,upper):
    #         slideSurf[i] =  pSlide[0]*xAxisSlide[i] + pSlide[1]
    
    # pSlide = np.polyfit(xAxisSlide,slideSurf,1)

        
        
    # plt.plot(xAxisSlide, slideSurf,'rx')
    # plt.plot(xAxisSlide, pSlide[0]*xAxisSlide + pSlide[1],'b')
    
    
    
    # xfull = np.arange(0,np.size(meanIntdB,1),1)
    # yGlass = pGlass[0]*xfull + pGlass[1]
    # ySlide = pSlide[0]*xfull + pSlide[1]
    
    # plt.plot(xfull,yGlass,'g')
    # plt.plot(xfull,ySlide,'y')
    
    # pixelThick = np.abs(yGlass-ySlide)
    
    # geoThickAnglemm = (np.abs(pGlass[1]-pSlide[1]) / np.sqrt(np.mean((pGlass[0],pSlide[0]))**2 + 1))* (1/100)
    
    # geoThickmm = pixelThick/100
    
    # print('mean = {}'.format(np.mean(geoThickAnglemm)))
    # print('std = {}'.format(np.std(geoThickmm)))
    
    
    # ThicknessAllmm = np.append(ThicknessAllmm,np.mean(geoThickAnglemm))
    
    


        