from getDiams_MFC import MFC as MFC
import os
import numpy as np
import Silo 

# set these variables
workingDir = "/p/work1/rebshan/" # where to loook for cases
caseCat = "M2B0P0C300"
postProcFolder = "/silo_hdf5/" # where data is stored within the case
nProc = 128
threshold = 0.1

## FIX : need to find dt and mesh density from the case.py file

timeStep = 1e-6 # not used?
meshDensity =  0.002/300

# initialize MFC class
MFC = MFC(postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep,nProc=nProc,threshold=threshold)
os.chdir(workingDir)

# grab the cases you want to analyze
case_list = [d for d in os.listdir() if d.startswith(caseCat) and os.path.isdir(d)] # grab all files in dir that start with string
case_numbers = []
#case_list = {'M2B100_200c_p10.NARWHAL'} # only one case for testing
print(f"case_list: {case_list}")
printThreshold = 10*threshold

header = ["timeStep","horizontal", "vertical","equator", "center_of_mass", "leading_edge","leading_edge_equator"]

diam_info_list = []

for caseName in case_list:
    caseFolder = workingDir + caseName + MFC.postProcFolder
    print(caseFolder)
    try:
        # Compute and collect diameter information across all time snapshots
        diameter_info = MFC.process_folder_diameter(caseFolder,caseName)
        diam_info_list.append(diameter_info)
        fName = "results_" + caseName + ".csv"

        diameter_info.to_csv(f"{caseFolder}/out_{caseName}_alpha{printThreshold}.csv",columns=header)
    except TypeError:
        print(f"folder {postProcFolder} returned an empty list")
        os.chdir("../")
        continue
 
