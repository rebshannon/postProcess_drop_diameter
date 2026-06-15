from getDiams_OF_pv import OpenFOAM_pv as OpenFOAMpv
import os
import numpy as np
#import Silo 

# set these variables
workingDir = "/p/work1/rebshan/OF_shockDrop/" # where to loook for cases
#workingDir = "/projectnb/aeracous/REBECCA/postProcessing/testingDirs/" # cases to look for
print(f"beginning now in {workingDir}")
caseCat = "M2B50OF"
postProcFolder = "/postProcess_alpha" # where data is stored within the case
timeStep = 9.1046e-7
meshDensity =  10e-6 #0.00127/300
threshold = 0.1

# initialize OF class
OF = OpenFOAMpv( postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep,threshold=threshold)
os.chdir(workingDir)

# grab the cases you want to analyze
case_list = [d for d in os.listdir() if d.startswith(caseCat) and os.path.isdir(d)] # grab all files in dir that start with string
case_numbers = []
#case_list = {'U267_D2_B1','U267_D2_B2','U267_D2_B2b','U267_D2_B3','U267_D2_B4','U267_D2_B45','U267_D2_B7','U267_D2_B10','U267_D2_B11','U267_D2_B12} # only one case for testing
print(f"case_list: {case_list}")
printThreshold = 10*threshold

header = ["times","horizontal", "vertical","equator", "center_of_mass", "leading_edge","leading_edge_equator"]
perim_header = ["timeStep",'perimeter', 'scale_factor', 'x_range', 'y_range', 'image_height', 'image_width']

diam_info_list = []
# perimeter_info_list = []
# post_shock_velocity_info_list = []

for caseName in case_list:
    caseFolder = workingDir + caseName + OF.postProcFolder
    try:
        # Compute and collect diameter information across all time snapshots
        diameter_info, perimeter_info = OF.process_folder_diameter(caseFolder,caseName)
        diam_info_list.append(diameter_info)
        fName = "results_" + caseName + ".csv"

        diameter_info.to_csv(f"{caseFolder}/out_{caseName}_alpha{printThreshold}.csv",columns=header)
        perimeter_info.to_csv(f"{caseFolder}/out_perim_{caseName}_alpha{printThreshold}.csv",columns=perim_header)
    except TypeError:
        print(f"folder {postProcFolder} returned an empty list")
        os.chdir("../")
        continue
 
