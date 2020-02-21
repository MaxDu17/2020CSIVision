import math

class CSIFrame(): #this object stores the csi frame information
    MAC = ""
    amplitude = []
    phase = []
    def __init__(self, MAC, amplitude, phase):
        self.MAC = MAC
        self.amplitude = amplitude
        self.phase = phase


class Parser():
    filter = ""
    filterstat = False
    def __init__(self, filterstat, filter):
            self.filter = filter
            self.filterstat = filterstat

    def extract_and_store(self, input_raw):
        input = str(input_raw)
        byelement = []
        amplitude = []
        phase = []

        imaginary = []
        real = []
        macaddress = ""
        if "<CSI>" in input:
            raw = input.split("</len>")[1] #this takes the later half of the data
            raw = raw.split("\\")[0]
            byelement = raw.split()
            macaddress = input.split("<address>")[1]
            macaddress = macaddress.split("</address>")[0] #this should get the mac address
            if len(byelement) != 384: #corruption protection! 
                return None



            for i in range(len(byelement)):
                if i%2 == 0:
                        imaginary.append(int(byelement[i]))
                else:
                    real.append(int(byelement[i]))

            for i in range(int(len(byelement)/2)):
                amplitude_val = math.sqrt(imaginary[i]**2 + real[i]**2)
                phase_val = math.atan2(imaginary[i], real[i])
                amplitude.append(amplitude_val)
                phase.append(phase_val)

            object = CSIFrame(macaddress, amplitude, phase)

            return object

        return None






