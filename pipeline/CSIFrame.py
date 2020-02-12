class CSIFrame(): #this object stores the csi frame information
    MAC = ""
    amplitude = []
    phase = []
    def __init__(self, MAC, amplitude, phase):
        self.MAC = MAC
        self.amplitude = amplitude
        self.phase = phase