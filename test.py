import tensorflow as tf
import numpy as np
import os
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import pickle

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filename = np.array(pickle.load(open('filename.pkl','rb')))

print(feature_list.shape)
print(filename.shape)

model = ResNet50(weights = 'imagenet', include_top = False)
model.trainable = False
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    model,
    GlobalMaxPooling2D()
])

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

test_feature = extract_features('.\sample\showes_image.png', model)


# for finding the nearest features**********************************************
from sklearn.neighbors import NearestNeighbors
def find_nearest_neighbors_sklearn(array_list, target_vector, n):
  """
  Finds the n nearest neighbors' positions using scikit-learn (efficient for larger datasets).

  Args:
      array_list: A NumPy array of multidimensional vectors.
      target_vector: The target vector.
      n: The number of nearest neighbors to find.

  Returns:
      A list of the n nearest neighbor positions (indices).
  """
  # Convert array list to NumPy array (recommended for scikit-learn)
  X = array_list

  # Create a NearestNeighbors object (metric='euclidean' by default)
  nbrs = NearestNeighbors(n_neighbors=n)

  # Fit the data (builds internal data structures)
  nbrs.fit(X)

  # Query for nearest neighbors
  distances, indices = nbrs.kneighbors(np.array([target_vector]))

  # Flatten the results (assuming single target vector query)
  distances = distances.flatten()
  indices = indices.flatten()

  return indices

# Sample array (replace with your actual data)

nearest_positions = find_nearest_neighbors_sklearn(feature_list, test_feature, 6)
nearest_image_path = []
for pos in nearest_positions:
   nearest_image_path.append(filename[pos])

import cv2


# Create windows for each image (adjust window titles as needed)
for i, image_path in enumerate(nearest_image_path):
  img = cv2.imread(image_path)
  cv2.imshow(f"Image {i+1}", img)

# Wait for a key press to close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# ***************************************************************************************