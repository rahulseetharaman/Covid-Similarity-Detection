# import the necessary packages
from pyimagesearch.similarity import GradCAMS
from os import walk
import os
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

import shutil
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import time
from scipy import spatial
import imutils
import cv2
# construct the argument parser and parse the arguments

from tensorflow.keras.models import load_model
model=load_model("pneumothorax.h5")
model_t = load_model("covid19.h5")
# load the original image from disk (in OpenCV format) and then
# resize the image to its target dimensions

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

f1 = []

for (dirpath, dirnames, filenames) in walk("E:\\covidclassifier\\test\\covid\\1"):
    f1.extend(filenames)
    break

sim = 0
count = 0
for file in f1:
    count+=1
    print(count)
    try:
        shutil.copy("E:\\covidclassifier\\test\\covid\\1\\"+file,"E:\\covidclassifier\\test\\test\\1\\")
    except:
        time.sleep(3)

    test_generator = datagen.flow_from_directory("E:\\covidclassifier\\test\\test\\")
    orig = cv2.imread("E:\\covidclassifier\\test\\test\\1\\"+file)
    x, y = test_generator.next()
    image = x[0]
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    preds = model.predict_generator(test_generator)

    i = np.argmax(preds[0])
    cam = GradCAMS(model, i)
    cam_t = GradCAMS(model_t, i)
    val1 = cam.compute_heatmap(image)
    val_t = cam_t.compute_heatmap(image)

    a_sparse, b_sparse = sparse.csr_matrix(val1), sparse.csr_matrix(val_t)

    sim_sparse = cosine_similarity(a_sparse, b_sparse, dense_output=False)
    # print(sim_sparse)
    s = 0
    for i in range(8):
        for j in range(8):
            s+=sim_sparse[i,j]
    s = s/64
    print(s)
    sim+=s
    try:
        os.remove("E:\\covidclassifier\\test\\test\\1\\"+file)
    except:
        time.sleep(3)
        os.remove("E:\\covidclassifier\\test\\test\\1\\" + file)
print(sim/count)









