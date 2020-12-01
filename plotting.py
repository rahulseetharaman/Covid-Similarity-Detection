import matplotlib.pyplot as plt
# accarr = [0.9038,0.9671,0.9640,0.9752,0.9826,0.9677,0.9777,0.9758,0.9901,0.9864,0.9876,0.9882,0.9907,0.9870,0.9895,0.9864,0.9888,0.9839,0.9820,0.9820,0.9777,0.9801,0.9864,0.9888,0.9895,0.9944,0.9901,0.9870,0.9963,0.9938,0.9919,0.9950,0.9857,0.9913,0.9919,0.9926,0.9938,0.9932,0.9957,0.9932,0.9963,0.9888,0.9932,0.9833,0.9895,0.9814,0.9833,0.9801,0.9950,0.9895]
# plt.plot(accarr)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.show()
# lossarr = [0.7946,0.2679,0.2543,0.2405,0.2163,0.2696,0.2339,0.2257,0.1878,0.2045,0.1838,0.1776,0.1758,0.1731,0.1637,0.1733,0.1645,0.1817,0.1727,0.1677,0.1930,0.1822,0.1756,0.1448,0.1450,0.1302,0.1359,0.1441,0.1235,0.1212,0.1220,0.1141,0.1342,0.1327,0.1264,0.1149,0.1184,0.1153,0.1027,0.1013,0.1018,0.1229,0.1005,0.1410,0.1116,0.1308,0.1264,0.1570,0.0959,0.1147]
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

test_generator1=datagen.flow_from_directory("E:\\covidclassifier\\test\\covidtrue")
test_generator2=datagen.flow_from_directory("E:\\covidclassifier\\test\\covidfalse")

true = [1 for i in range(0,460)]
false = [0 for i in range(0,460)]

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
# models = ["atelectasis","consolidation","effusion","emphysema","fibrosis_r","infiltration","mass","nodule","pleural_thickening","pneumonia","pneumothorax"]
models = ["infiltration","mass","nodule","pleural_thickening","pneumonia","pneumothorax"]
# models = ["fibrosis"]
import numpy as np
from sklearn.metrics import auc
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
for i in models:
    model = load_model(i+".h5")
    print(i)
    result1 = model.predict_generator(test_generator1)
    result2 = model.predict_generator(test_generator2)
    # print(len(result))
    # # Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size + 1)
    y_pred1 = np.argmax(result1, axis=1)
    y_pred2 = np.argmax(result2, axis=1)
    # print(y_pred)
    # print('Confusion Matrix')
    # print(confusion_matrix(test_generator.classes, y_pred))
    # print(test_generator.classes)

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(true, y_pred1)
    tp = tpr_keras
    print(tpr_keras)

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(false, y_pred2)
    fp = fpr_keras
    print(fpr_keras)
    auc_rf = auc(fp, tp)
    plt.plot(fp,tp,label=i+"(area = {:.3f})".format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(i+".png")