from getDiams import MFC as MFC
import os
import numpy as np
import Silo 

# set these variables
workingDir = "/projectnb/aeracous/REBECCA/MFC/v5.0.6/shockDropParam/2D/" # where to loook for cases
caseCat = "case"
postProcFolder = "/silo_hdf5/" # where data is stored within the case
nProc = 32

## FIX : need to find dt and mesh density from the case.py file

timeStep = 1e-6 # not used?
meshDensity =  0.0035/300

# initialize MFC class
MFC = MFC(postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep,nProc=nProc)
os.chdir(workingDir)

# grab the cases you want to analyze
#case_list = [d for d in os.listdir() if d.startswith(caseCat) and os.path.isdir(d)] # grab all files in dir that start with string
case_numbers = []
case_list = {'case1'} # only one case for testing
print(f"case_list: {case_list}")

header = ["timeStep","horizontal", "vertical","equator", "center_of_mass"]

diam_info_list = []

for caseName in case_list:
    caseFolder = workingDir + caseName + MFC.postProcFolder
    print(caseFolder)
    try:
        # Compute and collect diameter information across all time snapshots
        diameter_info = MFC.process_folder_diameter(caseFolder,caseName)
        diam_info_list.append(diameter_info)
        fName = "results_" + caseName + ".csv"

        diameter_info.to_csv(f"{caseFolder}/out_{caseName}.csv",columns=header)
    except TypeError:
        print(f"folder {postProcFolder} returned an empty list")
        os.chdir("../")
        continue
 
