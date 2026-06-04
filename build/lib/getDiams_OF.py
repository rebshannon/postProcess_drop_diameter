from getDiams import postProcess

import os
import re
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import time
from scipy import ndimage
import csv
import fluidfoam

class OpenFOAM(postProcess):
    def __init__(self,postProcFolder,meshDensity,timeStep):
        """pattern : str
            Regular expression for filename matching (not path).
        """
        super().__init__(postProcFolder,meshDensity)
        
        self.timeStep = timeStep

        self.pattern = r'cellCenterData_\d+\.csv' # what the data is saved under

        # header names used for tree
        self.alphaVar = 'alpha.water'
        self.x = 'cellCenterCoords:0'
        self.y = 'cellCenterCoords:1'
        self.z = 'cellCenterCoords:2'
    # file formatting stuff

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
        
        # get list of times 
        try:
            timeList = pd.read_csv('timeList.csv',header=None,dtype=str).to_numpy()
        except FileNotFoundError:
            print(f"Time List file does not exsit. Run 'foamListTimes > timeList.csv' on {folder}")
        
        if timeList: # if there's data
 
            # load mesh
            
            times = []
            time_strs = []
            diameters = np.array([0,0,0,0,0,0])
            for time in timeList:

                #  
                data, times, time_strs = self.load_dataframe(file,True,times,time_strs)
                coords = self.get_water_points(data)
                horizontal_diameter, vertical_diameter, leading_edge = self.calculate_diameters(coords)
                equator_diameter, leading_edge_equator = self.calculate_equator_diameter(coords)
                center_of_mass_diameter = self.calculate_centOfMass_diameter(coords)

                diameters = np.vstack([diameters, [horizontal_diameter, vertical_diameter,equator_diameter,center_of_mass_diameter, leading_edge, leading_edge_equator]])


            times = pd.DataFrame(times, columns=[caseName])
            time_strs = pd.DataFrame(time_strs,columns=[caseName])

            diameters = np.delete(diameters, (0), axis=0)
            
            diameter_info = pd.DataFrame()
            diameter_info["times"] = times
            # diameter_info["a_pca"] = diameters[:,0]
            # diameter_info["b_pca"] = diameters[:,1]
            # diameter_info["a"] = diameters[:,2]
            # diameter_info["b"] = diameters[:,3]
            diameter_info["horizontal"] = diameters[:,0]
            diameter_info["vertical"] = diameters[:,1]
            diameter_info["equator"] = diameters[:,2]
            diameter_info['center_of_mass'] = diameters[:,3]
            diameter_info['leading_edge'] = diameters[:,4]
            diameter_info['leading_edge_equator'] = diameters[:,5]
            os.chdir("../")
            print(f"case:{caseName}")
            print(f"diameter_info:{diameter_info}")
            #print(f"mach_no: {mach_no}")
            return diameter_info #, mach_no

    def load_dataframe(self,file,getData, times,time_strs):
        """Load CSVs into DataFrames and extract times from filenames.

        Parameters
        ----------
        file_list : list[str]
            Paths to CSV files to load.
        folder : str
            Name of the Mach folder; becomes the column name for the times table.
        getData : bool
            When True, actually read CSVs; when False, only parse and return times.

        Returns
        -------
        tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]
            - dataframes: list of loaded DataFrames (empty if getData is False)
            - times: single-column DataFrame of float times labeled by folder
            - time_strs: single-column DataFrame of string times labeled by folder
        """
        dataframe = []

        timeStepNum = self.get_time_from_fileName(file)
        intTime = timeStepNum * self.timeStep
        time_strs.append(intTime)
        intTime = float(intTime)
        times.append(intTime)

        if(getData==True):
            df = pd.read_csv(file)
                        
            #print('reading elapsed')
            #elapsed = time.time() -t
            #print(elapsed)

        print('data files loaded')
        return df, times, time_strs
