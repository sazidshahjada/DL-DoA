import numpy as np

class CATARCSModel:
    def __init__(self, M, N, d=0.5, F=2):
        self.M = M
        self.N = N
        self.d = d  
        self.F = F  
        self.M_prime = self.M // self.F  
        self.L = self.M + self.N

        # Number of physical sensor = 2M + N - 1
        self.number_of_sensors = 2 * self.M + self.N - 1

        # DOF = (4MN - 1) / 2
        self.dof = (4 * self.M * self.N - 1) // 2

    def sensor_positions(self):
        # Subarray 1: {0, -M_0, -2M_0, ..., -(N-1)M_0}d
        grid_S1 = [-n * self.M_prime * self.d for n in range(0, self.N)]
        
        # Subarray 2: {0 - (N-1)M_0 + L, N - (N-1)M_0 + L, ... }d
        offset = -(self.N - 1) * self.M_prime + self.L
        grid_S2 = [(m * self.N + offset) * self.d for m in range(0, 2 * self.M - 1)]
        
        # Combine and sort unique positions
        grid_S = list(set(grid_S1 + grid_S2))  
        grid_S.sort()  
        S = grid_S
        # S = [pos * self.d for pos in grid_S]
        
        return S




class GoldenRatioModel:
    def __init__(self, M, N, d=1):
        self.M = M
        self.N = N
        self.d = d
        self.phi = (1 + np.sqrt(5)) / 2
        self.number_of_sensors = self.M + self.N + 4
        self.dof = int(np.round(self.M * self.N + self.M * (self.phi ** self.N + self.N)))

    def sensor_positions(self):
        # Subarray 1: {0, M, 2M, ..., NM}
        H1 = [n * self.N for n in range(0, self.M + 1)]
        H1 = list(set(H1))

        # Subarray 2: {M * N + M * (phi^i + i) for i in 1 to N}
        H2 = [self.M * self.N + self.M * (self.phi ** i + i) for i in range(1, self.N + 1)]
        H2 = np.round(H2).astype(int)
        H2 = list(set(H2))

        # Subarray 3: {M * phi, N * phi, (M * N) / phi}
        H3 = [self.M * self.phi, self.N * self.phi, (self.M * self.N) / self.phi]
        H3 = np.round(H3).astype(int)
        H3 = list(set(H3))

        # Final sensor positions
        S = list(set(H1 + H2 + H3))
        S.sort()
        S = [int(x) for x in S]

        return S
    



# Testing
if __name__ == "__main__":
    M = 6
    N = 7
    d = 0.5

    catarcs_model = CATARCSModel(M, N, d)
    print("CATARCS Sensor Positions:", catarcs_model.sensor_positions())
    print("CATARCS Number of Sensors:", catarcs_model.number_of_sensors)
    print("CATARCS DOF:", catarcs_model.dof)

    # golden_ratio_model = GoldenRatioModel(M, N, d)
    # print("Golden Ratio Sensor Positions:", golden_ratio_model.sensor_positions())
    # print("Golden Ratio DOF:", golden_ratio_model.dof)