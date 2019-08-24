"""
bearing_som.ipynb
Created at 08/06/2019
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sompy.sompy import SOMFactory

def normalize(dataframe):
    result = dataframe.copy()
    for feature_name in dataframe.columns:
        max_value = dataframe[feature_name].max()
        min_value = dataframe[feature_name].min()
        result[feature_name] = (dataframe[feature_name] - min_value) / (max_value - min_value)
    return result

# read the data
trainingDataframe = pd.read_csv("../Dataset/TrainingSet.csv")

# have a peek of the data
print(trainingDataframe.head())

# normalize the data
trainingDataframe = normalize(trainingDataframe)
print(trainingDataframe.head())

# split the data
condition_0 = trainingDataframe[:256]
condition_1 = trainingDataframe[256: 512]
condition_2 = trainingDataframe[512: 768]
condition_3 = trainingDataframe[768: 1024]
condition_4 = trainingDataframe[1024: 1280]
condition_5 = trainingDataframe[1280: 1536]
condition_6 = trainingDataframe[1536: 1792]
condition_7 = trainingDataframe[1792:]

# training dataset
cond_0_training = condition_0.head(192)
cond_1_training = condition_1.head(192)
cond_2_training = condition_2.head(192)
cond_3_training = condition_3.head(192)
cond_4_training = condition_4.head(192)
cond_5_training = condition_5.head(192)
cond_6_training = condition_6.head(192)
cond_7_training = condition_7.head(192)

frames = [cond_0_training,
          cond_1_training,
          cond_2_training,
          cond_3_training,
          cond_4_training,
          cond_5_training,
          cond_6_training,
          cond_7_training]
training_split_0 = pd.concat(frames)

# validation dataset
cond_0_validation = condition_0.tail(64)
cond_1_validation = condition_1.tail(64)
cond_2_validation = condition_2.tail(64)
cond_3_validation = condition_3.tail(64)
cond_4_validation = condition_4.tail(64)
cond_5_validation = condition_5.tail(64)
cond_6_validation = condition_6.tail(64)
cond_7_validation = condition_7.tail(64)

frames = [cond_0_validation,
          cond_1_validation,
          cond_2_validation,
          cond_3_validation,
          cond_4_validation,
          cond_5_validation,
          cond_6_validation,
          cond_7_validation]
validation_split_0 = pd.concat(frames)

# shuffle the data
training_split_0 = shuffle(training_split_0)
validation_split_0 = shuffle(validation_split_0)

# reset the index after shuffle
training_split_0.reset_index(inplace=True, drop=True) 
validation_split_0.reset_index(inplace=True, drop=True)


print(training_split_0.head())


# Train the data
mapSize = [20, 20]

sm = SOMFactory().build(training_split_0.values, mapSize, normalization = "var", lattice = "rect", initialization = "random", component_names=training_split_0.columns)
sm.train(n_job=1, verbose = None, train_rough_len=2, train_finetune_len = 100)    # I left some of the codes as the example provided


# plot the results, components map
from sompy.visualization.mapview import View2D

view2D = View2D(20, 20, "", text_size=12)
view2D.show(sm, col_sz = 3, which_dim = "all", denormalize = False)

# Hit maps
from sompy.visualization.bmuhits import BmuHitsView

vhts = BmuHitsView(15, 10, "Hits Map", text_size = 12)
vhts.show(sm, anotate = False, onlyzeros = False, labelsize = 12, cmap = "jet", logaritmic = False)

# U martix
from sompy.visualization.umatrix import UMatrixView

u = UMatrixView(15,15, 'umatrix', show_axis=True, text_size = 12, show_text=False)
UMAT = u.build_u_matrix(sm, distance=1, row_normalized=False)
u.show(sm)