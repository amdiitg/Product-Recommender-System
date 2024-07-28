import tensorflow as tf
import numpy as np
import os
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
import pickle


model = ResNet50(weights = 'imagenet', include_top = False)
model.trainable = False
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    model,
    GlobalMaxPooling2D()
])
# print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path, target_size = (224,224))
    img_array = image.img_to_array(img)
    img_expanded = np.expand_dims(img_array, axis = 0)

    img_preprocessed = preprocess_input(img_expanded)

    result = model.predict(img_preprocessed).flatten()
    normalised_result = result/norm(result)
    # print(img_features)
    # print(img_features.shape)
    return normalised_result

file_name = []
for file in os.listdir('images'):
    file_name.append(os.path.join('images',file))

feature_list = []

for file in tqdm(file_name):
    feature = extract_features(file,model)
    feature_list.append(feature)

pickle.dump(feature_list,open('embeddings.pkl','wb'))  
pickle.dump(file_name,open('filename.pkl','wb'))