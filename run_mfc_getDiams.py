# ====== IMPORT ===== #
from getDiams_MFC import MFC as MFC
import os
import numpy as np
import Silo 

# ====== INITIALIZE ===== #

# CASE INFO
workingDir = "/p/home/rebshan/MFC/examples/" # where to loook for cases
postProcFolder = "/silo_hdf5/" # where data is stored within the case
nProc = 128 # number of processors used in simulation
threshold = 0.1 # alpha threshold - assumes water is 1
timeStep = 1e-6 
meshDensity = 9.6e-4

# POST PROCESS OPTIONS
getDiameter = True
getPerimeter = True

# CHOOSE CASES
case_list = {'2D_shockdroplet'} # choose exact cases
#caseCat = "M2B" # search based on case name
#case_list = [d for d in os.listdir() if d.startswith(caseCat) and os.path.isdir(d)] # grab all files in dir that start with string

print(f"case_list: {case_list}")

# ====== RUN POST PROCESS ===== #

# initialize MFC class
MFC = MFC(postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep,nProc=nProc,threshold=threshold)
os.chdir(workingDir)

for caseName in case_list:
    caseFolder = workingDir + caseName + postProcFolder
    try:
        MFC.run_post_process(caseFolder, caseName, calcDiam=getDiameter, calcPerim=getPerimeter)
    except TypeError:
        print(f"folder {postProcFolder} returned an empty list")
        os.chdir("../")
        continue
 
