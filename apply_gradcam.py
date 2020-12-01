# import the necessary packages
from pyimagesearch.gradcam import GradCAM

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np

import imutils
import cv2
# construct the argument parser and parse the arguments

from tensorflow.keras.models import load_model
model=load_model("fibrosis_r.h5")
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

test_generator=datagen.flow_from_directory("E:\\covidclassifier\\test\\test\\")


orig = cv2.imread("E:\\covidclassifier\\test\\test\\1\\covidFibrosis.jpg")

print(orig)
#resized = cv2.resize(orig, (256, 256))
# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it

#image = load_img("E:\\covidclassifier\\test\\covid\\1\\1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-000-fig1b.png", target_size=(256, 256))
x,y = test_generator.next()

image = x[0]
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# use the network to make predictions on the input image and find
# the class label index with the largest corresponding probability
preds = model.predict_generator(test_generator)
# print(preds)
i = np.argmax(preds[0])
# decode the ImageNet predictions to obtain the human-readable label

# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)
# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
# draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, "1", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.8, (255, 255, 255), 2)
# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)