class DataPipeline():
    data = []
    label = ""
    oneHot = []
    startIndex = 0
    def __init__(self, data, label, oneHot, startIndex):
        self.data = data
        self.label = label
        self.oneHot = oneHot
        self.startIndex = startIndex