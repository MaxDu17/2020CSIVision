import matplotlib
import pickle
import numpy as np

from pipeline.DataParser import DataParser
k = DataParser()
k.load_data("BedroomAmbient")

k.plot(k.get_square_data_norm(0, 4))
k.plot(k.get_square_data_norm(0, 3))
k.plot(k.get_square_data_norm(0, 2))
k.plot(k.get_square_data_norm(0, 1))
k.plot(k.get_square_data_norm(0, 0))
