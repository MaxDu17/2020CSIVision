from pipeline.DatasetMaker_Universal import DatasetMaker_Universal
from pipeline.ProjectUtility import Utility
from pipeline.DataParser_Universal  import DataParser_Universal
from pipeline.DataPipeline import DataPipeline
import numpy as np

DP  = DataParser_Universal()
DM = DatasetMaker_Universal(DP, 40, 50)
tool = Utility()

data, label = DM.next_epoch_batch()
print("\n\n")
print("The validation set start at:" + str(DM.validation_start) + "and constains " +
      str(DM._debug_get_valid_size()) + " elements")
print("The test set starts at: " + str(DM.test_start) + " and contains " +
      str(DM._debug_get_test_size()) + " elements")
print("The training set starts at (universal): " + str(DM.train_start))

print("This is the number of frames for training: " + str(DM._debug_get_train_size()))
print(np.shape(data))
#print(data)
print(np.shape(label))
#print(label)
