from getDiams import OpenFOAM_pv as OpenFOAMpv
import os
import numpy as np
#import Silo 

# set these variables
workingDir = "/projectnb/aeracous/REBECCA/DOD_CAVSYM/" # where to loook for cases
#workingDir = "/projectnb/aeracous/REBECCA/postProcessing/testingDirs/" # cases to look for
caseCat = "U267_D2_B"
postProcFolder = "/postProcessing/pvData" # where data is stored within the case
timeStep = 1e-6
meshDensity =  2e-6 #2e-6 #0.00127/300

# initialize OF class
OF = OpenFOAMpv( postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep)
os.chdir(workingDir)

# grab the cases you want to analyze
#case_list = [d for d in os.listdir() if d.startswith(caseCat) and os.path.isdir(d)] # grab all files in dir that start with string
case_numbers = []
case_list = {'U267_D2_B1'} # only one case for testing
print(f"case_list: {case_list}")

header = ["times","horizontal", "vertical","equator", "center_of_mass"]

diam_info_list = []
# perimeter_info_list = []
# post_shock_velocity_info_list = []

for caseName in case_list:
    caseFolder = workingDir + caseName + OF.postProcFolder
    try:
        # Compute and collect diameter information across all time snapshots
        diameter_info = OF.process_folder_diameter(caseFolder,caseName)
        print(diameter_info.shape)
        # perimeter_info, case_number = gsFns.process_folder_perimeter(postProcFolder)
        # post_shock_velocity_info, case_number = gdFns.process_folder_post_shock_velocity(postProcFolder)
        # perimeter_info_list.append(perimeter_info)
        diam_info_list.append(diameter_info)
        # post_shock_velocity_info_list.append(post_shock_velocity_info)
        #case_numbers.append(case_number)
        fName = "results_" + caseName + ".csv"

        diameter_info.to_csv(f"{caseFolder}/out_{caseName}_alpha09.csv",columns=header)
    except TypeError:
        print(f"folder {postProcFolder} returned an empty list")
        os.chdir("../")
        continue
 