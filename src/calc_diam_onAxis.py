def calculate_equator_diameters(self,df):
        """Estimate diameters using 2D axis-aligned extents of the main droplet.

        Steps
        -----
        - Threshold water volume fraction to segment droplet cells.
        - Build a KD-tree and find connected components within a small radius.
        - Keep the largest connected component as the main droplet (rejects spray).
        - Compute axis-aligned extents in X and Y as diameter proxies.

        Returns
        -------
        tuple[float, float]
            (horizontal_diameter, vertical_diameter) where horizontal is span in X, and vertical is
            set to twice the span in Y (historical heuristic kept intact).
        """
        #Load the data from the file
        data = df
        #t = time.time()
        
        # Filter out the points that belong to the water droplet
        water_points = data[data[self.alphaVar] > self.threshold]
        #print('water points found')

        # Get the coordinates of the water points
        coords = water_points[[self.x, self.y, self.z]].values

        if len(coords) == 0:
            return 0, 0  # No water points found

        # check which points correspond to points along y-axis
        coords_hAxis = coords[coords[:,1]<OF.meshDensity]

        # Build a KD-tree for fast neighbor search
        tree = cKDTree(coords_hAxis)
        #print('tree made')

        # Find neighbors within a small distance (e.g., 2x grid spacing)
        # You may need to adjust this radius based on your data resolution
        radius = 2*self.meshDensity

        adjacency = tree.query_ball_tree(tree, r=radius)
        #print('tree ball ran')

        # Build connected components using BFS
        visited = np.zeros(len(coords_hAxis), dtype=bool)
        components = []
        for i in range(len(coords_hAxis)):
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
        coords_hAxis = coords_hAxis[largest_component]
        # fig, ax = plt.subplots()
        # ax.scatter(coords[:, 0], coords[:, 1], s=1)
        # ax.set_aspect('equal', 'box')
        # plt.savefig('../droplet_scatter.png')
        # plt.close(fig)

        # Calculate the distances between all pairs of points
        max_x = coords_hAxis[:, 0].max()
        min_x = coords_hAxis[:, 0].min()

        equator_diameter = max_x - min_x

        
        return equator_diameter