from getDiams_OF import OpenFOAM as OpenFOAM
import os
import numpy as np

# set these variables
workingDir = "/p/work1/rebshan/" # where to loook for cases
caseCat = "DS6_MULES.N"
timeStep = 1e-6
meshDensity =  0.00127/400
threshold = 0.9

# initialize OF class
OF = OpenFOAM(meshDensity=meshDensity,timeStep=timeStep,threshold=threshold,postProcFolder='None')
os.chdir(workingDir)

# grab the cases you want to analyze
#case_list = [d for d in os.listdir() if d.startswith(caseCat) and os.path.isdir(d)] # grab all files in dir that start with string
case_numbers = []
case_list = {'DS6_MULES.NARWHAL','DS6_SOLVE.NARWHAL','fixedDS6.NARWHAL'} # only one case for testing

print(f"case_list: {case_list}")

header = ["times","horizontal", "vertical","equator", "center_of_mass", "leading_edge","leading_edge_equator"]

diam_info_list = []
printThreshold = 10*threshold

for caseName in case_list:
    caseFolder = workingDir + caseName 
    try:
        # Compute and collect diameter information across all time snapshots
        diameter_info = OF.process_folder_diameter(caseFolder,caseName)
        print(diameter_info.shape)
        diam_info_list.append(diameter_info)
        fName = "results_" + caseName + ".csv"

        diameter_info.to_csv(f"{caseFolder}/out_{caseName}_alpha{printThreshold}.csv",columns=header)
    except TypeError:
        print(f"folder {caseFolder} returned an empty list")
        os.chdir("../")
        continue
 
