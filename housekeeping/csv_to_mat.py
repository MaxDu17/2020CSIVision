
#this code take a confusion matrix csv and plots it
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from pipeline.ProjectUtility import Utility
from pipeline.Hyperparameters import Hyperparameters

HYP = Hyperparameters()
Util = Utility()
version = "AllDataCNN_" + HYP.MODE_OF_LEARNING

base_directory = "../Graphs_and_Results/Vanilla" + "/" + version + "/confusion.csv"

test = open(base_directory, "r")
matrix = Util.cast_csv_to_float(test)

df_cm = pd.DataFrame(matrix, range(5), range(5))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()


'''

fig=plt.figure()
ax = fig.add_subplot(111) #111 is an argument in the form of 3 numbers
color = ax.matshow(matrix)
fig.colorbar(color)
plt.show()
'''

