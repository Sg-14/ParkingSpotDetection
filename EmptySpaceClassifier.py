import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import  SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  train_test_split
from skimage.io import  imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import os

# Preparing the images data
input_dir = 'parking-image-data'
categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

print(data)
print(labels)

# Training and testing data split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

# Model Training
SVMClassifier = SVC()

# Hyperparameter Tuning
param_grid = [{
    'gamma': [0.01, 0.001, 0.0001, 0.00001],
    'C': [0.1, 1, 10, 100, 1000]
}]

grid_search_cv = GridSearchCV(SVMClassifier, param_grid)

grid_search_cv.fit(X_train, y_train)

# Prediction and Model Training
best_model = grid_search_cv.best_estimator_
y_pred = best_model.predict(X_test)

score = accuracy_score(y_pred, y_test)
print('The accuracy of the model is : {}%'.format(str(score*100)))

print(classification_report(y_test, y_pred, target_names=['empty', 'not_empty']))

# Load the test image and resize it
path = 'test_image.png'
img = imread(path)
img_resize = resize(img, (15, 15, 3))

# Flatten the resized test image
img_flattened = img_resize.flatten()

# Reshape the flattened image to match the shape expected by the model
img_flattened_reshaped = img_flattened.reshape(1, -1)

# Make prediction using the best model
prediction = best_model.predict(img_flattened_reshaped)

# Display the result
print("The predicted image is:", categories[prediction[0]])


#  Saving the final model
pickle.dump(best_model, open('./model.p', 'wb'))