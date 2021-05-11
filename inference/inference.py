# import the packages
import numpy as np
import os
import pandas
import seaborn
import matplotlib.pyplot as plt
from tensorflow import keras

# load the model
model = keras.models.load_model('../input/modelpart/pnue_model.h5')

# strings
label = {
    0 : 'Patient has a Normal report.',
    1 : 'Patient suffers from Pneumonia.',
}

# predict
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def predict (img_path) :
    
    # load image
    # convert to array
    X_test = []
    y_test = []
    for types in ['NORMAL/', 'PNEUMONIA/'] :
        for path in os.listdir(img_path+types) :
            
            image = load_img(img_path + types + path, color_mode
                             ='grayscale',target_size=(128,128))
            
            X_test.append(img_to_array(image))
            if types == 'PNEUMONIA/' :
                y_test.append(1)
            else:
                y_test.append(0)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_test/= 255.0
    
    # predict
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1)
    y_pred = np.around (y_pred)
    
    y_pred = np.array(y_pred, dtype = int)
    
    return [label[y_pred[i]] for i in range(y_pred.shape[0])]

# COMMENTED
# img_path = '../input/chest-xray-pneumonia/chest_xray/test/'
# result = predict(img_path)