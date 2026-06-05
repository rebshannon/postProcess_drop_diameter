from getDiams import postProcess
import os
import re
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import time
from scipy import ndimage
import Silo
import csv

class MFC(postProcess):

    # file formatting stuff
    def __init__(self,postProcFolder,meshDensity,nProc,timeStep,threshold):
        super().__init__(postProcFolder,meshDensity,threshold)
        
        self.nProc = nProc
        self.timeStep = timeStep
        self.pattern = r'collection_'
 
        # header names used for tree
        self.alphaVar = 'a'
        self.x = 'x'
        self.y = 'y'
        self.z = 'z'

    def process_folder_diameter(self,folder,caseName):
        """Process one case folder to compute a/b diameters over time.

        Parameters
        ----------
        folder : str
            Path of the caseCat folder to enter and analyze.

        Returns
        -------
        tuple[pd.DataFrame, float] | None
            (diameter_info, mach_no) if files are found; otherwise None.

        diameter_info has columns:
        - times: snapshot times (float)
        - a_pca, b_pca: diameters from PCA method
        - a, b: diameters from axis-aligned extent method
        """
        try:
            os.chdir(folder)
        except FileNotFoundError:
            print(f"no folder with name {folder}")
            return None
        
        fName = folder + 'root'
        matching_files = self.find_matching_files(fName) # look for files named like pattern - gives path to all files

        if matching_files: # if there's data
       
            sorted_matching_files=sorted(matching_files,key=lambda x: float(re.split(r'[_,.]+',x)[-2])) # sorted based on time; assumes the timestep is the last thing listed

            #get data and a list of times files were saved at
            times = []
            time_strs = []
            diameters = np.array([0,0,0,0,0,0])
            for file in sorted_matching_files:
                #data, times,time_strs = self.load_dataframes(file, caseName,True)
                tStep = self.get_time_from_fileName(file)
                times.append(tStep)
                time_strs.append(tStep)
                data = self.extract_and_combine_data(file,tStep,folder)
                
                coords = self.get_water_points(data)
                horizontal_diameter, vertical_diameter, leading_edge = self.calculate_diameters(coords)
                equator_diameter, leading_edge_equator = self.calculate_equator_diameter(coords)
                center_of_mass_diameter = self.calculate_centOfMass_diameter(coords)
                diameters = np.vstack([diameters, [horizontal_diameter, vertical_diameter,equator_diameter,center_of_mass_diameter, leading_edge, leading_edge_equator]])

            times = pd.DataFrame(times, columns=[caseName])
            time_strs = pd.DataFrame(time_strs,columns=[caseName])

            diameters = np.delete(diameters, (0), axis=0)
            
            diameter_info = pd.DataFrame()
            diameter_info["timeStep"] = times
            diameter_info["horizontal"] = diameters[:,0]
            diameter_info["vertical"] = diameters[:,1]
            diameter_info["equator"] = diameters[:,2]
            diameter_info['center_of_mass'] = diameters[:,3]
            diameter_info['leading_edge'] = diameters[:,4]
            diameter_info['leading_edge_equator'] = diameters[:,5]
            os.chdir("../")
            print(f"case:{caseName}")
            print(f"diameter_info:{diameter_info}")
            return diameter_info 

    def extract_and_combine_data(self,file,tStep,folder):
        root_DB = Silo.Open(file,Silo.DB_READ)
        root_VarInfo = root_DB.GetVarInfo('alpha1',1)
        alphaExtents = root_VarInfo['extents']

        ## FIX: assumes alpha = 1 is water

        waterProc = []
        for i in range(0,2*self.nProc-1,2):
            if alphaExtents[i] >= self.threshold or alphaExtents[i+1] >= self.threshold:
                waterProc.append(i/2)

        alphaList = []
        for proc in waterProc:

            # read processor database
            fName = folder+'p' + str(int(proc)) + '/' + str(int(tStep)) + '.silo'
            proc_DB = Silo.Open(fName,Silo.DB_READ)

            # take out alpha and mesh info
            alphaDict = proc_DB.GetVarInfo('alpha1',1)
            alphaValues = alphaDict['value0']

            meshDict = proc_DB.GetVarInfo('rectilinear_grid',1)
            xcoordVal = meshDict['coord0']
            ycoordVal = meshDict['coord1']

            # interpolate mesh to cell centers
            # rectilinear grid is given as grid points, want to find the cell centers
            xCellCent = []
            yCellCent = []
            for ind, val in enumerate(xcoordVal):
                if ind == len(xcoordVal) - 1:
                    break
                xCellCent.append((val + xcoordVal[ind+1]) /2)

            for ind, val in enumerate(ycoordVal):
                if ind == len(ycoordVal) - 1:
                    break
                yCellCent.append((val + ycoordVal[ind+1]) /2)

            # FIX: only works in 2D
            # make list that [x,y,z,alpha]
            indA = 0
            for indY,valY in enumerate(yCellCent):
                for indX, valX in enumerate(xCellCent):
                    alphaList.append([valX,valY,0,alphaValues[indA]])
                    indA += 1
                    

            proc_DB.Close()
        root_DB.Close()
        
        # with open(f"out_case1_{tStep}.csv",mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(alphaList)

        alpha_df = pd.DataFrame(alphaList)
        alpha_df.columns = ['x','y','z','a']
       

        return alpha_df