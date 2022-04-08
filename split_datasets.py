import os
import numpy as np
import shutil

root_dir = "cut/datasets/horse2zebra"
old_A = os.path.join(root_dir, "TrainA")
old_B = os.path.join(root_dir, "TrainB")

new_dir = "cut/datasets/horse2zebra_split"
new_A = os.path.join(new_dir, "TrainA")
new_B = os.path.join(new_dir, "TrainB")

# make directories
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
if not os.path.exists(new_A):
    os.makedirs(new_A)
if not os.path.exists(new_B):
    os.makedirs(new_B)

# randomly sample 100 images from each old dataset
allFileNames = os.listdir(old_A)
np.random.shuffle(allFileNames)
allFileNames = allFileNames[:100]
for fileName in allFileNames:
    shutil.copy(os.path.join(old_A, fileName), os.path.join(new_A, fileName))

allFileNames = os.listdir(oldB)
np.random.shuffle(allFileNames)
allFileNames = allFileNames[:100]
for fileName in allFileNames:
    shutil.copy(os.path.join(old_B, fileName), os.path.join(new_B, fileName))
