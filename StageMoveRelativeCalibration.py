# %%
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
folderLocation = r'H:/PhD/OCT Data/Compression Test/Calibration/12 October/Step 0.05'                                                           
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

#%%
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


# %%

parentDirectory = 'H:/PhD/OCT Data/Compression Test'

force = np.array((0,1100,2094,3085,4453,5444))*9.81 /1000 

N=6 #Number of measurements per trial


loadCellFiles = glob.glob(parentDirectory + '/ForceCalib_*.txt')

for files in enumerate(loadCellFiles):
    print(files)
    loadCellData = np.loadtxt(files[1])
    
    if files[0] == 0:
        voltageRatio = loadCellData
    
    voltageRatio = np.vstack((voltageRatio,loadCellData))


plt.figure()
for i in range(len(voltageRatio)):
    plt.plot(voltageRatio[i,:],force,'x')



polyCoeffForce = np.polyfit(np.mean(voltageRatio,0),force,1)

xpointsVoltage = np.arange(min(np.mean(voltageRatio,0)),max(np.mean(voltageRatio,0)), max(np.mean(voltageRatio,0))/1000)
forceFitted = polyCoeffForce[0]*xpointsVoltage + polyCoeffForce[1]

#plt.plot(np.mean(voltageRatio, axis=0),force,'k--')
plt.plot(xpointsVoltage,forceFitted,'k--')
plt.ylabel('Weight (N)')
plt.xlabel('Voltage Ratio (mV/V)')
plt.grid()

    
# %%

parentDirectory = "H:/PhD/OCT Data/Compression Test/Calibration/PDMS/Trial 9 - 10 steps 0.05"

voltageData = np.loadtxt(glob.glob(parentDirectory + '/*.txt')[0])

forceData = polyCoeffForce[0]*voltageData + polyCoeffForce[1] # Measured in N

displacement = np.arange(0,0.5,0.05) # measured in mm


#
OCTFilesCh0 = sorted(glob.glob(parentDirectory + '/Ch0*.tdms'))
OCTFilesCh1 = sorted(glob.glob(parentDirectory + '/Ch1*.tdms'))
peakPosition =np.array(())
for allImages in OCTFilesCh0:
    print(allImages)
    variableName = allImages.split('/')[-1]
    variableName = variableName.split('_',1)[-1]
    variableName = variableName.rsplit('.',1)[0]
    Int = []
    Ret = []
    Ch0Complex = []
    Ch1Complex = []
    t1 = time.time()
    Int,Ret, Ch0Complex,Ch1Complex = directory.loadData(parentDirectory,npy,output,variableName,A_scan_num,B_scan_num,padSize)
    t2 = time.time()
    print('It took {:.2f} seconds to load in the data'.format(t2-t1))

    sliced=10
    intdB = 10*np.log10(Int)

    meanIntdB = np.mean(intdB,0)
    meanIntdB = np.mean(meanIntdB,1)
    
    plt.figure(1)
    plt.plot(meanIntdB)
    start=10
    peak = np.argmax(meanIntdB[int(start*scale):len(meanIntdB)])+start*scale
    peakPosition = np.append(peakPosition,peak)
    plt.plot(peak, meanIntdB[int(peak)],'x')


peakPositionmm = peakPosition /(100*scale)
mmShift = np.abs(peakPositionmm - np.max(peakPositionmm))
plt.figure()
plt.plot(displacement,mmShift,'rx')
plt.xlabel('translation stage displacement (mm)')
plt.ylabel('cavity compression (mm)')
plt.title('Cavity compression vs translation stage measureements - ideally should be gradient 1')
plt.grid()
plt.tight_layout()


xpointCompression = np.linspace(min(mmShift),max(mmShift),1000)
polyCoeffCavity = np.polyfit(mmShift, forceData,1)
FittedCavity = polyCoeffCavity[0]*xpointCompression + polyCoeffCavity[1]
plt.figure()
plt.plot(mmShift, forceData, 'rx')
plt.plot(xpointCompression,FittedCavity, 'b--')
plt.xlabel('Cavity Compression (mm)')
plt.ylabel('Force (N)')
plt.title('Cavity compression vs measured force - Needs to be linear')
plt.grid()
plt.tight_layout()

areaCavity = 40.60 *40.6 -  (np.pi*13**2 + 4 * np.pi * (6.05/2)**2) #mm^2
thickCavity = 12 #mm

stressCavity = max(forceData)/(areaCavity*10**-6)
strainCavity = (max(mmShift)/1000)/(12/1000)
modulus = (10**-6)*stressCavity/strainCavity

