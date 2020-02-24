import matplotlib
import pickle
import numpy as np
from models.Utility import Utility
tool = Utility()

class CSIFrame(): #this object stores the csi frame information
    MAC = ""
    amplitude = []
    phase = []
    def __init__(self, MAC, amplitude, phase):
        self.MAC = MAC
        self.amplitude = amplitude
        self.phase = phase

def normalizeandround(frame):
    maxval = max(frame)
    minval = min(frame)

    for i in range(len(frame)):
        frame[i] = int(255 * (frame[i]-minval)/(maxval-minval))
    return frame

def normalizeandround2(matrix):
    minval = np.min(matrix)
    maxval = np.max(matrix)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = int(255 * (matrix[i][j] - minval) / (maxval - minval))
    return matrix
frames = pickle.load(open("../datasets/BedroomAmbient/BedroomAmbient.pkl", "rb"))
matrix = []
for frame in frames:
    if(frame != None):
        matrix.append(normalizeandround(frame.amplitude[1:]))
        print(frame.amplitude[122:135])
#matrix = [p[66:122] for p in matrix]
#matrix = [p[1:] for p in matrix]
matrix = np.transpose(np.asarray(matrix))
#matrix = normalizeandround2(matrix)
print(matrix.shape)
#print(matrix[60:125][18])
tool.display_image(matrix)
