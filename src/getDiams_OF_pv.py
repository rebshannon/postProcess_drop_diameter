from getDiams import postProcess
import os
import re
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import time
from scipy import ndimage
import fluidfoam
import csv

class OpenFOAM_pv(postProcess):

    def __init__(self,postProcFolder,meshDensity,timeStep,threshold):
        """pattern : str
            Regular expression for filename matching (not path).
        """
        super().__init__(postProcFolder,meshDensity,threshold)
        
        self.timeStep = timeStep

        #PV CHANGE: cellCetnerData
        self.pattern = r'cellCenterData_\d+\.csv' # what the data is saved under

        # header names used for tree
        #PV CHANGE: cellCetnerCoords
        self.alphaVar = 'alpha.water'
        self.x = 'cellCenterCoords:0'
        self.y = 'cellCenterCoords:1'
        self.z = 'cellCenterCoords:2'
    # file formatting stuff

    def process_folder_diameter(self,folder,caseName,calcDiam,calcPerim):
        """Process one case folder to compute a/b diameters over time.

        Parameters
        ----------
        folder : str
            Path of the caseCat folder to enter and analyze.
        caseName : [str]
            name of the case to be analyzed
        calcDiam : [bool]
            turn on diameter calculation
        calcPerim : [bool]
            turn on perimeter calculation

        Returns
        -------
        tuple[pd.DataFrame, float] | None
            (diameter_info, mach_no) if files are found; otherwise None.

        diameter_info has columns:
        - timeStep number
        - horizontal: max(x) - min(x)
        - vertical: 2*max(y)
        - equator: [max(x) - min(x)] | y = 0
        - center_of_max: [2*max(y)] | x = center
        - leading_edge: min(x)
        - leading_edge_equator: min(x) | y = 0

        perimeter_info has columns:
        - timeStep number
        - perimeter
        - scale_factor
        - x_range
        - y_range
        - image_height
        - image_width
        """
        try:
            os.chdir(folder)
        except FileNotFoundError:
            print(f"no folder with name {folder}")
            return None
       
        matching_files = self.find_matching_files(folder) # look for files named like pattern - gives path to all files
       
        if matching_files: # if there's data
       
            sorted_matching_files=sorted(matching_files,key=lambda x: float(re.split(r'[_,.]+',x)[-2])) # sorted based on time; assumes the timestep is the last thing listed

            #get data and a list of times files were saved at
            times = []
            time_strs = []
            diameters = np.array([0,0,0,0,0,0])
            perimeters = np.array([0,0,0,0,0,0])

            for file in sorted_matching_files:

                data, times, time_strs = self.load_dataframe(file,True,times,time_strs)
                coords = self.get_water_points(data)

                if calcDiam:
                    # calc diameters
                    horizontal_diameter, vertical_diameter, leading_edge = self.calculate_diameters(coords)
                    equator_diameter, leading_edge_equator = self.calculate_equator_diameter(coords)
                    center_of_mass_diameter = self.calculate_centOfMass_diameter(coords)

                    # combine into array
                    diameters = np.vstack([diameters, [horizontal_diameter, vertical_diameter,equator_diameter,center_of_mass_diameter, leading_edge, leading_edge_equator]])
                
                if calcPerim:
                
                    # calc perim and add to array
                    perimeter, contour, scale_factor, x_range, y_range, image_height, image_width = self.calculate_perimeter(coords)         
                    perimeters = np.vstack([perimeters, [perimeter, scale_factor,x_range,y_range,image_height,image_width]])

            times = pd.DataFrame(times, columns=[caseName])
            time_strs = pd.DataFrame(time_strs,columns=[caseName])

            diameter_info = pd.DataFrame()
            perimeter_info = pd.DataFrame()
            
            if calcDiam:
                diameters = np.delete(diameters, (0), axis=0)
                diameter_info["timeStep"] = times
                diameter_info["horizontal"] = diameters[:,0]
                diameter_info["vertical"] = diameters[:,1]
                diameter_info["equator"] = diameters[:,2]
                diameter_info['center_of_mass'] = diameters[:,3]
                diameter_info['leading_edge'] = diameters[:,4]
                diameter_info['leading_edge_equator'] = diameters[:,5]

            if calcPerim:
                perimeters = np.delete(perimeters,(0),axis=0)
                perimeter_info = pd.DataFrame()
                perimeter_info["timeStep"] = times
                perimeter_info["perimeter"] = perimeters[:,0]
                perimeter_info["scale_factor"] = perimeters[:,1]
                perimeter_info["x_range"] = perimeters[:,2]
                perimeter_info["y_range"] = perimeters[:,3]
                perimeter_info["image_height"] = perimeters[:,4]
                perimeter_info["image_width"] = perimeters[:,5]

            os.chdir("../")
            print(f"case:{caseName}")
            return diameter_info, perimeter_info

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

            return df, times, time_strs
