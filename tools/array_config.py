import sys
sys.path.insert(0, '/home/sajid/Work/DL-DoA')

import numpy as np
import matplotlib.pyplot as plt
from tools.coarray_models import CATARCSModel, GoldenRatioModel


class Coprime_Array:
    """
    Coprime Array Class

    This class creates a coprime array based on the selected model and
    provides utilities for analyzing its coarray properties.
    """

    def __init__(self, M, N, d, model_type='CATARCS'):

        self.M = M
        self.N = N
        self.d = d

        # Select array model
        if model_type == 'CATARCS':
            self.model = CATARCSModel(self.M, self.N, self.d)

        elif model_type == 'GoldenRatio':
            self.model = GoldenRatioModel(self.M, self.N, self.d)

        else:
            raise ValueError("Invalid model type. Choose 'CATARCS' or 'GoldenRatio'.")

        # Extract sensor positions
        self.sensor_positions = self.model.sensor_positions()

        # Metadata
        self.number_of_sensors = self.model.number_of_sensors
        self.dof = self.model.dof

        # Virtual array and hole positions
        self.virtual_positions = self.virtual_array_positions()
        self.hole_positions = self.get_hole_positions()
        self.uniform_positions = self.equivalent_uniform_array()


    def virtual_array_positions(self):
        """
        Compute the positive difference coarray (virtual array).

        The virtual array is formed using pairwise differences
        between sensor positions, keeping only non-negative lags.

        Returns
        -------
        numpy array
            Sorted unique positive virtual array positions.
        """

        sensor_positions = np.array(self.sensor_positions)

        # Pairwise difference matrix
        diff_matrix = sensor_positions[:, None] - sensor_positions[None, :]

        # Keep only non-negative lags
        positive_lags = diff_matrix[diff_matrix >= 0]

        # Remove duplicates and sort
        virtual_positions = np.sort(np.unique(positive_lags))

        return virtual_positions


    def get_hole_positions(self):
        """
        Detect holes in the positive virtual array.

        Holes are missing positions in the expected uniform
        virtual array grid.

        Returns
        -------
        numpy array
            Positions of holes in the virtual array.
        """

        virtual_positions = np.round(self.virtual_array_positions(), 3)

        # Determine grid resolution automatically
        diffs = np.diff(np.sort(virtual_positions))
        resolution = np.min(diffs)

        max_pos = np.max(virtual_positions)

        # Expected uniform grid
        expected_positions = np.round(
            np.arange(0, max_pos + resolution, resolution), 3
        )

        # Find missing positions
        hole_positions = [
            pos for pos in expected_positions
            if pos not in virtual_positions
        ]

        return np.array(hole_positions)
    

    def equivalent_uniform_array(self):
        """
        Compute the equivalent uniform array positions.

        This is the set of positions that would be occupied by a
        uniform linear array with the same aperture as the virtual array.

        Returns
        -------
        numpy array
            Positions of the equivalent uniform array.
        """

        virtual_positions = self.virtual_array_positions()

        # Determine grid resolution automatically
        diffs = np.diff(np.sort(virtual_positions))
        resolution = np.min(diffs)

        max_pos = np.max(virtual_positions)

        # Expected uniform grid
        uniform_positions = np.round(
            np.arange(0, max_pos + resolution, resolution), 3
        )

        return uniform_positions


    def visualize_array(self):
        """
        Visualize the array geometry.

        Displays:
        - Sensor positions
        - Virtual array positions
        - Hole locations
        """

        plt.figure(figsize=(20, 9))

        # Sensor positions
        plt.scatter(
            self.sensor_positions,
            np.ones_like(self.sensor_positions) * 2,
            color='green',
            s=100,
            label='Sensors'
        )

        # Virtual array
        virtual_positions = self.virtual_array_positions()

        plt.scatter(
            virtual_positions,
            np.ones_like(virtual_positions),
            color='blue',
            s=100,
            label='Virtual Array'
        )

        # Hole positions
        hole_positions = self.get_hole_positions()

        plt.scatter(
            hole_positions,
            np.zeros_like(hole_positions),
            color='red',
            s=100,
            label='Holes'
        )

        # Determine spacing automatically
        all_positions = np.concatenate([
            self.sensor_positions,
            virtual_positions
        ])

        diffs = np.diff(np.sort(np.unique(all_positions)))
        resolution = np.min(diffs)

        xmin = np.floor(np.min(all_positions))
        xmax = np.ceil(np.max(all_positions))

        xticks = np.arange(xmin, xmax + resolution, resolution)

        plt.xticks(xticks)

        plt.title(f'{self.model.__class__.__name__} Geometry', fontsize=16)

        plt.xlabel('Position', fontsize=14)
        plt.yticks([0, 1, 2], ['Holes', 'Virtual Array', 'Physical Array'], fontsize=12)

        plt.grid(True, which='major', linestyle='--', alpha=0.5)

        plt.show()



# Main execution
if __name__ == "__main__":
    M = 4
    N = 5
    d = 0.5

    print("=== CATARCS Model ===")
    catarcs_array = Coprime_Array(M=M, N=N, d=d, model_type='CATARCS')
    print(f"Sensor Positions: {catarcs_array.sensor_positions}")
    print(f"Number of Sensors: {catarcs_array.number_of_sensors}")
    print(f"Degrees of Freedom (DoF): {catarcs_array.dof}")
    print(f"Virtual Array Positions: {catarcs_array.virtual_positions}")
    print(f"Hole Positions: {catarcs_array.hole_positions}")  
    
    catarcs_array.visualize_array()