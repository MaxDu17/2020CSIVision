#this code take a confusion matrix csv and plots it
import matplotlib.pyplot as plt
from pipeline.ProjectUtility import Utility
from pipeline.Hyperparameters import Hyperparameters

HYP = Hyperparameters()
Util = Utility()
version = "DownstairsCNN_" + HYP.MODE_OF_LEARNING

base_directory = "../Graphs_and_Results/Vanilla" + "/" + version + "/confusion.csv"

test = open(base_directory, "r")
matrix = Util.cast_csv_to_float(test)

fig=plt.figure()
ax = fig.add_subplot(111) #111 is an argument in the form of 3 numbers
color = ax.matshow(matrix)
fig.colorbar(color)
plt.show()

