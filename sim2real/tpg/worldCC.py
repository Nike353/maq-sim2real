from typing import Union
import pdb
import numpy as np

class XYThetaTimeSolution:
    def __init__(self, xythetas: np.ndarray, times: np.ndarray, cost: float):
        assert len(xythetas) > 0
        self.xythetas = xythetas
        self.times = times
        self.cost = cost
        
    def get_xythetas(self) -> np.ndarray:
        return self.xythetas
    
    def get_times(self) -> np.ndarray:
        return self.times
    
    def get_cost(self) -> float:
        return self.cost
    
    def get_next_waypoint(self, bottom_time: float, upper_time: float):
        """
        bottom_time: float, Time of the bottom of the interval
        upper_time: float, Time of the top of the interval
        Returns:
            (3,): XYTheta of the next waypoint that is > bottom_time and <= upper_time
            float: time of the next waypoint that is > bottom_time and <= upper_time
        Example: if times is [0, 15, 20]
            If bottom_time = 10, upper_time = 23, then the next waypoint correponds to [15]
            If bottom_time = 10, upper_time = 14, then the next waypoint correponds to [14]
        """
        assert bottom_time <= upper_time
        assert bottom_time >= 0.0
        assert bottom_time <= self.times[-1]
        assert upper_time <= self.times[-1]
        
        # Find the index of the bottom and top times
        bottom_idx = np.searchsorted(self.times, bottom_time, side='right')
        # print(f"bottom_idx: {bottom_idx}, bottom_time: {bottom_time}, upper_time: {upper_time}")
        # print(f"self.times: {self.times}")
        if bottom_idx == len(self.times): # If bottom_time is the last time
            return self.xythetas[-1], self.times[-1]
        if self.times[bottom_idx] <= upper_time: # If the bottom_idx
            return self.xythetas[bottom_idx], self.times[bottom_idx]
        else:
            return getXYThetaAtTimes(self.xythetas, self.times, [upper_time])[0], upper_time
    
    def __repr__(self) -> str:
        # return f"XYThetaTimeSolution(xythetas.shape={self.xythetas.shape}, times.shape={self.times.shape})"
        return f"XYThetaTimeSolution(xythetas={self.xythetas}, times={self.times}, cost={self.cost})"
    
    
def getXYThetaAtTimes(xythetas: np.ndarray, times, query_times) -> np.ndarray:
    """Gets the XYTheta at the query times.
    Args:
        xythetas (...,3): XYTheta of robot
        times (N,): np.ndarray or array-like, Times of robot
        query_times (Q,): np.ndarray or array-like, Query times
    Returns:
        (Q,3): XYTheta at the query times
    """
    xVals = np.interp(query_times, times, xythetas[:,0]) # (Q,)
    yVals = np.interp(query_times, times, xythetas[:,1]) # (Q,)
    thetaVals = np.interp(query_times, times, xythetas[:,2]) # (Q,)
    return np.column_stack((xVals, yVals, thetaVals)) # (Q,3)


def load_instance_from_file(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        lines = f.readlines()
    
    ### Parse map
    first_line = lines[0].strip()
    assert len(first_line.split()) == 3 and first_line.split()[0] == "map"
    num_rows = int(first_line.split()[1])
    num_cols = int(first_line.split()[2])
    
    # Read map
    grid = np.array([list(line.rstrip()) for line in lines[1:num_rows+1]])
    bool_grid = np.zeros((num_rows, num_cols), dtype=bool)
    bool_grid[grid == '.'] = False
    bool_grid[grid == '@'] = True
    bool_grid[grid == 'T'] = True
    
    return bool_grid

def parse_map_file(file_path: str):
    """Parse a .map file into a numpy array. \n
    Args:
        file_path: Path to the .map file
    Returns:
        np.ndarray: 2D boolean array where True represents obstacles
    """
    with open(file_path) as f:
        # Read header
        version = f.readline().strip()
        assert version == "version search-zoo-v1", "Invalid map file format"
        
        resolution = f.readline().strip()
        assert resolution.split()[0] == "resolution", "Invalid map file format"
        resolution = float(resolution.split()[1]) # E.g. resolution 0.1
        
        next_line = f.readline().strip()
        assert next_line.split()[0] == "num_rows", "Invalid map file format"
        num_rows = int(next_line.split()[1]) # E.g. num_rows 32
        
        next_line = f.readline().strip()
        assert next_line.split()[0] == "num_cols", "Invalid map file format"
        num_cols = int(next_line.split()[1]) # E.g. num_cols 32
        
        # Read map data
        map_data = np.array([list(line.rstrip()) for line in f])
        
    # Reshape and convert to boolean
    occupancy_grid = np.zeros((num_rows, num_cols), dtype=bool)
    occupancy_grid[map_data == '.'] = False
    occupancy_grid[map_data == '@'] = True
    occupancy_grid[map_data == 'T'] = True
    return occupancy_grid, resolution