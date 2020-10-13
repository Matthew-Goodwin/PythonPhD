
import os,sys
file_dir = os.path.dirname('Z:/PhD/Python/GitHub/Universal')
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
folderLocation = r'Z:/PhD/OCT Data/Compression Test/Calibration/12 October/Step 0.04'                                                           
npy = r'\npy_files'                                                             
output = r'\output_files'                                                        
#Git Comment

# %%

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

    meanIntdB = np.mean(intdB,0)
    meanIntdB = np.mean(meanIntdB,1)
    
    plt.figure(1)
    plt.plot(meanIntdB)
    
    
    plt.figure(2)
    plt.clf()
    plt.plot(meanIntdB)
    plt.show()
    #peak = plt.ginput(2)
    #peak = np.array([i[0] for i in peak]).astype(int)
    peak = np.array((2500,10000))
    peakpos = np.append(peakpos,np.argmax(meanIntdB[peak[0]:peak[1]]) + peak[0])
peakpossorted = np.sort(peakpos)    
plt.figure()
plt.plot(peakpossorted,'bx')    

stepsize=[]
for i in range(1,len(peakpossorted)):
    stepsize = np.append(stepsize,(peakpossorted[i] - peakpossorted[i-1])/scale)
    
plt.figure()
plt.plot(stepsize,'bx')    
    
np.save(dataLocation +  output + '/stepsize_{}.npy'.format(sampleName),np.array(stepsize))
    
# %%
       
#Run thi section to plot the relative step size and the micrometer movement of the stage 
parentDirectory = 'H:/PhD/OCT Data/Compression Test/Calibration/12 October'

StepFileNames = sorted(glob.glob(parentDirectory + '/' + 'Step*'))

MedianStepSize =[]
RelStepSize = []

for StepSizeFile  in StepFileNames:
    StepSizeDist = StepSizeFile.split('\\')[-1]
    StepSizes = np.median(np.load(glob.glob(StepSizeFile + output + '/*npy')[0]))
    RelativeStepSize = float(StepSizeDist.split(' ')[-1])
    
    MedianStepSize = np.append(MedianStepSize, StepSizes)
    RelStepSize = np.append(RelStepSize, RelativeStepSize)
    
    



#Regression
MedianStepSizemicron = MedianStepSize*1000/100
xpoints = np.arange(min(RelStepSize), max(RelStepSize), ((max(RelStepSize)- min(RelStepSize))/100))
polyCoeff = np.polyfit(RelStepSize,MedianStepSizemicron,1)
yfitted = polyCoeff[0]*xpoints + polyCoeff[1]


plt.figure()
plt.plot(RelStepSize, MedianStepSizemicron, 'x')
plt.plot(xpoints,yfitted,'r-')
plt.xlabel('Relative Step Size')
plt.ylabel('Step Size (micrometer)')
plt.tight_layout()
plt.grid()
print('Relative step size needs to be multiplied by: {0:.2f} in order to convert the step to micrometers'.format(polyCoeff[0]))



    