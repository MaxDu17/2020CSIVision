from matplotlib import pyplot as plt
import pickle
import numpy as np

from pipeline.DataParser import DataParser
k = DataParser()
LOAD = "BedroomSleep"
k.load_data(LOAD)

def plot_operations(in_):
    array = k.frame_normalize_minmax_image(in_)
    matrix = np.transpose(np.asarray(array))
    images_plot = matrix.astype('uint8')
    plt.imshow(images_plot)
    return matrix

'''
#plt.plot(2, 3, 1, title = "Whole")
plot_operations(k.get_data(0, 1000, 4))
'''
START = 100

plt.subplot(2, 3, 1, title = "Whole")
plot_operations(k.get_square_data_norm(START, 4))


plt.subplot(2, 3, 2, title = "Third (Bottom)")
plot_operations(k.get_square_data_norm(START, 3))

plt.subplot(2, 3, 3, title = "Second (Middle)")
plot_operations(k.get_square_data_norm(START, 2))

plt.subplot(2, 3, 4, title = "First (Top)")
plot_operations(k.get_square_data_norm(START, 1))

plt.subplot(2, 3, 5, title = "Raw")
plot_operations(k.get_square_data_norm(START, 0))

plt.suptitle(LOAD)

plt.show(block=True)



'''
k.plot(k.get_square_data_norm(0, 4))
k.plot(k.get_square_data_norm(0, 3))
k.plot(k.get_square_data_norm(0, 2))
k.plot(k.get_square_data_norm(0, 1))
k.plot(k.get_square_data_norm(0, 0))
'''