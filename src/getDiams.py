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
import cv2
#from scipy.spactial import KDTree

class postProcess:

    def __init__(self, postProcFolder,meshDensity,threshold):
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

    def calculate_perimeter(self, coords):
        """Compute droplet perimeter by rasterizing into an image and contouring.

        Steps
        -----
        - Threshold 'Volume Fraction of Water' to isolate droplet cells.
        - Map (X, Y) coordinates to pixels with a scale factor (upsampling).
        - Apply morphological closing to glue small gaps between pixels.
        - Find the largest external contour and measure its perimeter.

        Returns
        -------
        tuple
            (perimeter, contour, scale_factor, x_range, y_range, H, W)
        """
        
        # Do this with image processing
        x_min, x_max = coords[:,0].min(), coords[:,0].max()
        y_min, y_max = coords[:,1].min(), coords[:,1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        scale_factor = 1e6  # Adjust scale for better resolution
        image_height = int(y_range * scale_factor) + 1
        image_width = int(x_range * scale_factor) + 1
        image = np.zeros((image_height, image_width), dtype=np.uint8)
        #print(int(y_range * scale_factor), int(x_range * scale_factor))
        for _, row in enumerate(coords):
            x_pixel = int((row[0] - x_min) * scale_factor)
            y_pixel = int((row[1] - y_min) * scale_factor)
            #if 0 <= x_pixel < image.shape[1] and 0 <= y_pixel < image.shape[0]:
            image[y_pixel, x_pixel] = 255

        # Use morphological closing to ensure white pixels are clumped together without losing resolution
        kernel = np.ones((30, 30), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite('droplet_image.png', image)

        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume the largest contour is the droplet outline
        contour = max(contours, key=cv2.contourArea)

        # Step 4: Calculate the perimeter of the droplet
        perimeter = cv2.arcLength(contour, True)
        return perimeter, contour, scale_factor, x_range, y_range, image_height, image_width
