# PythonPhD

File is made to hold the primary python scritps used to extract the data from the 1310nm Labview PS-OCT system.

**TDMSCodeUniversal.py**
Converts tdms files saved throught he labview acquisition to usuable float data

**TDMSProcessUniversal.py**
Takes the usable data and processes (fft + filtering) into intensity and retardation data.

**DirectorySetupUniversal.py**
Sets up directory with raw data and numpy and output files for saving data into.

**StageMoveRealtiveCalibration.py**
Used to detemrine the step size of the motorised stage.
