# ====== IMPORT ===== #
from getDiams_OF_pv import OpenFOAM_pv as OpenFOAMpv
import os
import numpy as np

# ====== INITIALIZE ===== #

# CASE INFO
workingDir = "/p/work1/rebshan/" # where to loook for cases
postProcFolder = "/postProcessing/pvData" # where data is stored within the case
threshold = 0.5 # alpha threshold - assumes water is 1
timeStep = 1e-6 
meshDensity =  2e-6

# POST PROCESS OPTIONS
getDiameter = True
getPerimeter = True

# CHOOSE CASES
case_list = {'U100B1.NARWHAL', 'U100B2.NARWHAL','U100B3.NARWHAL'} # choose exact cases
#caseCat = "U100B" # search based on case name
#case_list = [d for d in os.listdir() if d.startswith(caseCat) and os.path.isdir(d)] # grab all files in dir that start with string

print(f"case_list: {case_list}")

# ====== RUN POST PROCESS ===== #

# initialize MFC class
OF = OpenFOAMpv( postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep,threshold=threshold)
os.chdir(workingDir)

for caseName in case_list:
    caseFolder = workingDir + caseName + postProcFolder
    try:
        OF.run_post_process(caseFolder, caseName, calcDiam=getDiameter, calcPerim=getPerimeter)
    except TypeError:
        print(f"folder {postProcFolder} returned an empty list")
        os.chdir("../")
        continue
