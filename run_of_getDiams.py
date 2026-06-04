from getDiams_OF import OpenFOAM as OpenFOAM
import os
import numpy as np

# set these variables
workingDir = "/projectnb/aeracous/REBECCA/DOD_CAVSYM/" # where to loook for cases
print(f"beginning now in {workingDir}")
caseCat = "U267_D2_B"
postProcFolder = "" # where data is stored within the case
timeStep = 1e-6
meshDensity =  2e-6 #0.00127/300
threshold = 0.1

# initialize OF class
OF = OpenFOAMpv( postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep,threshold=threshold)
os.chdir(workingDir)

# grab the cases you want to analyze
case_list = [d for d in os.listdir() if d.startswith(caseCat) and os.path.isdir(d)] # grab all files in dir that start with string
case_numbers = []
#case_list = {'U267_D2_B1','U267_D2_B2','U267_D2_B2b','U267_D2_B3','U267_D2_B4','U267_D2_B45','U267_D2_B7','U267_D2_B10','U267_D2_B11','U267_D2_B12} # only one case for testing
print(f"case_list: {case_list}")

header = ["times","horizontal", "vertical","equator", "center_of_mass", "leading_edge","leading_edge_equator"]

diam_info_list = []

for caseName in case_list:
    caseFolder = workingDir + caseName + OF.postProcFolder
    try:
        # Compute and collect diameter information across all time snapshots
        diameter_info = OF.process_folder_diameter(caseFolder,caseName)
        print(diameter_info.shape)
        diam_info_list.append(diameter_info)
        fName = "results_" + caseName + ".csv"

        diameter_info.to_csv(f"{caseFolder}/out_{caseName}_alpha"{threshold}".csv",columns=header)
    except TypeError:
        print(f"folder {postProcFolder} returned an empty list")
        os.chdir("../")
        continue
 
