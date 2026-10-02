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
import glob

class OpenFOAM(postProcess):
    def __init__(self,meshDensity,timeStep,threshold,postProcFolder):
        """pattern : str
            Regular expression for filename matching (not path).
        """
        super().__init__(postProcFolder,meshDensity,threshold)
        
        self.timeStep = timeStep
        self.pattern = r'alphaCoords_\d+\.csv' # what the data is saved under
        

        # header names used for tree
        self.alphaVar = 'volumeFraction1'
        self.x = 'center:0'
        self.y = 'center:1'
        self.z = 'center:2'
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

        # zget list of processors
        processor_dirs = sorted(glob.glob(os.path.join(os.getcwd(), 'processor*')))
        if not processor_dirs:
            raise FileNotFoundError("No processor* directories found in case directory")
        
        if timeList.size > 0: # if there's data
 
            times = []
            time_strs = []
            diameters = np.array([0,0,0,0,0,0])
            for nStep, time in enumerate(timeList):

                time = time.item(0)
                times.append(time)
                time_strs.append(time)
                data = self.load_dataframe(time,processor_dirs,True)
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


    def load_dataframe(self, timestep, procList, parallel=True, verbose=False):
        """
        Read OpenFOAM alpha field and mesh coordinates from all valid processors.
        
        Parameters
        ----------
        timestep : float or int
            Timestep to read (e.g., 0.5, 1.0, etc.)
        parallel : bool, optional
            If True, read from processor directories (parallel); 
            if False, read reconstructed case (default: True)
        verbose : bool, optional
            If True, print progress messages (default: False)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: x, y, z, and {self.alphaVar}
            Only includes processors where max(alpha_field) > 0
        """

        # Initialize lists
        x_list = []
        y_list = []
        z_list = []
        alpha_list = []
        
        found_water = False
        
        # Loop over each processor
        for proc_dir in procList:
            if verbose:
                proc_name = os.path.basename(proc_dir) if parallel else "serial"
                print(f"Reading {proc_name}...")
            
            try:
                
                # Read alpha field
                try:
                    alpha_field = fluidfoam.readof.readfield(proc_dir, time_name=timestep, name=self.alphaVar)
                except KeyError:
                    raise ValueError(f"no {self.alphaVar} field")
                
                alpha_field = alpha_field.squeeze()
                
                # Check if processor has water
                if alpha_field.max() == 0:
                    if verbose:
                        print(f"  Skipping processor {proc_dir} (max alpha = 0)")
                    continue
                
                found_water = True
                
                # Read mesh
                x, y, z = fluidfoam.readof.readmesh(proc_dir)
                x = x.squeeze()
                y = y.squeeze()
                z = z.squeeze()
            
                # Append to lists
                x_list.extend(x)
                y_list.extend(y)
                z_list.extend(z)
                alpha_list.extend(alpha_field)
                
                if verbose:
                    print(f"  Added {len(x)} cells (max alpha = {alpha_field.max():.4f})")
            
            except ValueError as e:
                raise ValueError(str(e))
            except Exception as e:
                raise RuntimeError(f"Error reading from {os.path.basename(proc_dir)}: {e}")
        
        # Check if any water was found
        if not found_water:
            raise ValueError(f"no water for timestep {timestep}")
        
        # Build DataFrame
        df = pd.DataFrame({
            self.x: x_list,
            self.y: y_list,
            self.z: z_list,
            self.alphaVar: alpha_list
        })
        
        return df.astype('float32').reset_index(drop=True)

  