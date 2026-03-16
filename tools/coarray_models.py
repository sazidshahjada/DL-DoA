import numpy as np

class CATARCSModel:
    def __init__(self, M, N, d=0.5, F=2):
        self.M = M
        self.N = N
        self.d = d  # Inter-element spacing, default is λ/2, but used as 1 for integer positions
        self.F = F  # Compression factor
        self.M_prime = self.M / self.F  # Since M = F * M'
        # self.M_prime = 1
        self.L = self.M + self.N  # Translocation value
        self.number_of_sensors = 2 * self.M + self.N - 1
        self.dof = int(np.round(self.M * self.N + self.M * (self.M - 1) / 2))

    def sensor_positions(self):
        # Subarray 1: N sensors, starting at 0
        S1 = [-1 * n * self.M_prime * self.d for n in range(0, self.N)]    
        # Subarray 2: 2M-1 sensors, translocated by L
        S2 = [(m * self.N + self.L + S1[-1]) * self.d for m in range(0, 2 * self.M - 1)]
        
        # Combine and sort sensor positions
        S = list(set(S1 + S2))  # Use set to ensure no duplicates
        S.sort()  # Sort in ascending order
        
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
        # Layer 1
        H1 = [n * self.N for n in range(0, self.M + 1)]
        H1 = list(set(H1))
        # Layer 2
        H2 = [self.M * self.N + self.M * (self.phi ** i + i) for i in range(1, self.N + 1)]
        H2 = np.round(H2).astype(int)
        H2 = list(set(H2))
        # Layer 3
        H3 = [self.M * self.phi, self.N * self.phi, (self.M * self.N) / self.phi]
        H3 = np.round(H3).astype(int)
        H3 = list(set(H3))

        # Final sensor positions
        S = list(set(H1 + H2 + H3))
        S.sort()
        S = [int(x) for x in S]

        return S