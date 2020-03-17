
#this code take a confusion matrix csv and plots it
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import csv



def cast_csv_to_float(file_object):  # this takes a file object csv and returns a matrix
    logger = csv.reader(file_object)
    matrix = list(logger)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = float(matrix[i][j])
    return matrix



version = "AllDataCNN_third"

base_directory = "../Graphs_and_Results/Vanilla" + "/" + version + "/confusion.csv"

test = open(base_directory, "r")
matrix = cast_csv_to_float(test)

df_cm = pd.DataFrame(matrix, ["Ambient", "Fall", "Sleep", "Walk", "Work"], ["Ambient", "Fall", "Sleep", "Walk", "Work"])
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()


