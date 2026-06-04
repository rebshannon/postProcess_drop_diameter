import os
import re
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import time
from scipy import ndimage
import Silo
import csv
import fluidfoam
#from scipy.spactial import KDTree

class postProcess:

    def __init__(self, postProcFolder,meshDensity):
        self.postProcFolder = postProcFolder
        self.meshDensity = meshDensity
        self.threshold = threshold

    def find_matching_files(self,directory):
        """Recursively find files in a directory tree whose names match a regex.
        Used for OFPV and MFC to match cellCenterData and collection_

        Parameters
        ----------
        directory : str
            Root directory to walk.

        Returns
        -------
        list[str]
            List of absolute/normalized file paths that match the pattern.
        """
        matching_files = []
        regex = re.compile(self.pattern)
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if regex.match(filename):
                    matching_files.append(os.path.join(root, filename))
        return matching_files

    def get_time_from_fileName(self,fileName):
        """Read file name and extract time/timestep - the last number in the name

        Parameters
        ----------
        directory : fileName
            Name of file where timestep data is stored.

        Returns
        -------
        int[str]
            Time step number.
        """
        intertStep = fileName.split('_')[-1]
        tStep = float(intertStep.split('.')[0])
        return tStep
    
    def get_water_points(self,df):
        """Find coordinates that have water above defined threshold

        Steps
        -----
        - Threshold water volume fraction to segment droplet cells.
        - Threshold y-coordinates to segement droplet cells that are along the x axis (equator)
        
        Returns
        -------
        list[float, float]
            (coords, coords_hAxis) where coords is all of the water coordinate and 
            coords_hAxis is just those along the x axis
        """
        #Load the data from the file
        data = df
        
        # Filter out the points that belong to the water droplet
        water_points = data[data[self.alphaVar] > self.threshold]

        # Get the coordinates of the water points
        coords = water_points[[self.x, self.y, self.z,self.alphaVar]].values

        if len(coords) == 0:
            return 0  # No water points found     

        return coords

    def calculate_diameters(self,coords):
        """Estimate diameters using 2D axis-aligned extents of the main droplet.

        Steps
        -----
        - Build a KD-tree and find connected components within a small radius.
        - Keep the largest connected component as the main droplet (rejects spray).
        - Compute axis-aligned extents in X and Y as diameter proxies.

        Returns
        -------
        tuple[float, float]
            (horizontal_diameter, vertical_diameter) where horizontal is span in X, and vertical is
            set to twice the span in Y (historical heuristic kept intact).
        """ 

        # Build a KD-tree for fast neighbor search
        tree = cKDTree(coords[:,:3])

        # Find neighbors within a small distance (e.g., 2x grid spacing)
        # You may need to adjust this radius based on your data resolution
        radius = 2*self.meshDensity

        adjacency = tree.query_ball_tree(tree, r=radius)

        # Build connected components using BFS
        visited = np.zeros(len(coords), dtype=bool)
        components = []
        for i in range(len(coords)):
            if not visited[i]:
                queue = [i]
                component = []
                while queue:
                    idx = queue.pop()
                    if not visited[idx]:
                        visited[idx] = True
                        component.append(idx)
                        queue.extend([n for n in adjacency[idx] if not visited[n]])
                components.append(component)      
      
        # Keep only the largest component (assumed to be the main droplet)
        largest_component = max(components, key=len)
        coords = coords[largest_component]

        # Calculate the distances between all pairs of points
        max_x = coords[:, 0].max()
        min_x = coords[:, 0].min()
        max_y = coords[:, 1].max()
        min_y = coords[:, 1].min()

        #elapsed = time.time() -t
        #print(elapsed)
        
        # The major diameter is the maximum distance between any two points
        horizontal_diameter = max_x - min_x
        vertical_diameter = 2 * (max_y - min_y)  # Initial guess for vertical diameter
        leading_edge = min_x
        
        return horizontal_diameter, vertical_diameter, leading_edge
    
    def calculate_equator_diameter(self,coords):
        """Estimate equator diameter using center horizontal axis of the main droplet.

        Steps
        -----
        - Build a KD-tree and find connected components within a small radius.
        - Keep the largest connected component as the main droplet (rejects spray).
        - Compute axis-aligned extents on the x axis (droplet horizontal equator).

        Parameters
        ----------
        coords : list[str]
            coordinates that have water above threshold.
        Returns
        -------
        list[float]
            equator_diameter : horizontal diameter along droplet equator
	    leading_edge : location of leading edge of droplet on y = 0
        """
       
        # Build a KD-tree for fast neighbor search
        tree = cKDTree(coords[:,:3])

        # Find neighbors within a small distance (e.g., 2x grid spacing)
        # You may need to adjust this radius based on your data resolution
        radius = 2*self.meshDensity

        adjacency = tree.query_ball_tree(tree, r=radius)

        # Build connected components using BFS
        visited = np.zeros(len(coords), dtype=bool)
        components = []
        for i in range(len(coords)):
            if not visited[i]:
                queue = [i]
                component = []
                while queue:
                    idx = queue.pop()
                    if not visited[idx]:
                        visited[idx] = True
                        component.append(idx)
                        queue.extend([n for n in adjacency[idx] if not visited[n]])
                components.append(component)
        
        
        # Keep only the largest component (assumed to be the main droplet)
        largest_component = max(components, key=len)
        coords = coords[largest_component]

        # check which points correspond to points along y-axis
        coords_hAxis = coords[coords[:,1] < self.meshDensity]
        coords_hAxis = coords_hAxis[coords_hAxis[:,1] > -self.meshDensity]

        # Calculate the distances between all pairs of points
        max_x = coords_hAxis[:, 0].max()
        min_x = coords_hAxis[:, 0].min()

        equator_diameter = max_x - min_x
        leading_edge_eq = min_x

        return equator_diameter, leading_edge_eq

    def calculate_centOfMass_diameter(self,coords):
        """Estimate equator diameter using center horizontal axis of the main droplet.

        Steps
        -----
        - Build a KD-tree and find connected components within a small radius.
        - Keep the largest connected component as the main droplet (rejects spray).
        - Compute axis-aligned extents on the x axis (droplet horizontal equator).

        Parameters
        ----------
        coords : list[str]
            coordinates that have water above threshold.
        Returns
        -------
        list[float]
            center_of_mass_diameter : vertical diameter along droplet center of mass
        """
        # Build a KD-tree for fast neighbor search
        tree = cKDTree(coords[:,:3])

        # Find neighbors within a small distance (e.g., 2x grid spacing)
        # You may need to adjust this radius based on your data resolution
        radius = 2*self.meshDensity

        adjacency = tree.query_ball_tree(tree, r=radius)

        # Build connected components using BFS
        visited = np.zeros(len(coords), dtype=bool)
        components = []
        for i in range(len(coords)):
            if not visited[i]:
                queue = [i]
                component = []
                while queue:
                    idx = queue.pop()
                    if not visited[idx]:
                        visited[idx] = True
                        component.append(idx)
                        queue.extend([n for n in adjacency[idx] if not visited[n]])
                components.append(component)
        
        
        # Keep only the largest component (assumed to be the main droplet)
        largest_component = max(components, key=len)
        coords = coords[largest_component]

        # Find centroid of the largest component
        #center_coords = ndimage.center_of_mass(coords)
        center_coords = np.average(coords[:,:3],axis=0,weights=coords[:,3])
        
        # check which points correspond to points along the center of mass axis
        coords_vAxis = coords[(coords[:,0] - center_coords[0]) < self.meshDensity]
        coords_vAxis = coords_vAxis[(center_coords[0] - coords_vAxis[:,0]) > -self.meshDensity]

        # Calculate the distances between all pairs of points
        max_y = coords_vAxis[:, 1].max()
        min_y = coords_vAxis[:, 1].min()

        centroid_diameter = 2*(max_y - min_y)

        return centroid_diameter


class OpenFOAM_pv(postProcess):

    def __init__(self,postProcFolder,meshDensity,timeStep,threshold):
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
       
        matching_files = self.find_matching_files(folder) # look for files named like pattern - gives path to all files
        print(matching_files)
       
        if matching_files: # if there's data
       
            sorted_matching_files=sorted(matching_files,key=lambda x: float(re.split(r'[_,.]+',x)[-2])) # sorted based on time; assumes the timestep is the last thing listed
            
            #self.update_header_names(matching_files[0])

            #get data and a list of times files were saved at
            times = []
            time_strs = []
            diameters = np.array([0,0,0,0,0,0])
            for file in sorted_matching_files:
                #print('file = ',file)
                #data, times,time_strs = self.load_dataframes(file, caseName,True)
                data, times, time_strs = self.load_dataframe(file,True,times,time_strs)
                #print('times = ',times)
                #major_diameter_pca,minor_diameter_pca = calculate_diameters_pca(df)
                coords = self.get_water_points(data)
                #print('coords =', coords)
                horizontal_diameter, vertical_diameter, leading_edge = self.calculate_diameters(coords)
                equator_diameter, leading_edge_equator = self.calculate_equator_diameter(coords)
                center_of_mass_diameter = self.calculate_centOfMass_diameter(coords)
                #diameters = np.vstack([diameters, [major_diameter_pca, minor_diameter_pca, major_diameter, minor_diameter]])
                diameters = np.vstack([diameters, [horizontal_diameter, vertical_diameter,equator_diameter,center_of_mass_diameter, leading_edge, leading_edge_equator]])
                #print('diams found')

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
        print(times)
        return df, times, time_strs



