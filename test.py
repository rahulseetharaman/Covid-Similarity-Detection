from __future__ import print_function


from keras.models import load_model
model=load_model("pleural_thickening.h5")
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

test_generator=datagen.flow_from_directory("E:\\covidclassifier\\test\\covid")


result = model.predict_generator(test_generator)
print(result)
op = 0
prob = 0
count = 0
for i in result:
    if i[0]>i[1]:
        print(0)
        op = 0
        prob = i[0]
    else:
        print(1)
        op = 1
        prob = i[1]
        count+=1
    print(prob)
print(count/len(result)*100)