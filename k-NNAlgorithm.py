import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

# Function for reading coordinates from a file
def read_coordinates_from_file(filename):
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            coordinates.append((x, y, z))
    return coordinates

# Function for plotting coordinates
def plot_coordinates(coordinates, title, folder, file_name):
    fig = plt.figure()
    plot = fig.add_subplot(111, projection='3d')
    
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    z = [coord[2] for coord in coordinates]

    plot.scatter(x, y, z, c='r', marker='o')
    
    plot.set_xlabel('X-Axis')
    plot.set_ylabel('Y-Axis')
    plot.set_zlabel('Z-Axis')
    
    plot.set_title(title)

    # Save the plot in the folder with a file name
    nameIMG = f'plot_{folder}_{file_name}.jpg'
    fig.savefig(nameIMG, format='jpg')
    plt.close()

# Function for calculating the RMS
def calculate_rms(data):
    varSquared = [x**2 for x in data]
    varMean = np.mean(varSquared)
    rms = np.sqrt(varMean)
    return rms

# Function for creating a feature vector from coordinates
# 3 vectors 3*7
def get_feature_vector(coordinates):
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    z = [coord[2] for coord in coordinates]

    rms_x = calculate_rms(x)
    rms_y = calculate_rms(y)
    rms_z = calculate_rms(z)

    return [rms_x, rms_y, rms_z]

# Function for plotting the feature vectors in different colours
def plot_3d_data(data_list, train_labels):
    fig = plt.figure()
    plot = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b']  # R, G, B for three groups (walking, sitting, jogging)

    for idx, feature_vector in enumerate(data_list):
        xs = feature_vector[0]  #rms values
        ys = feature_vector[1]  
        zs = feature_vector[2] 

        group_color = colors[train_labels[idx] - 1]  # different group -> differenz color

        plot.scatter(xs, ys, zs, c=group_color, label=f'Dataset {idx+1}')

    plot.set_xlabel('RMS_x')
    plot.set_ylabel('RMS_y')
    plot.set_zlabel('RMS_z')

    fig.savefig('AllFeatures.jpg', format='jpg')
    plt.close


#Links to the target files
train_data_folder = 'data'
train_folders = ['act01', 'act02', 'act03']

train_feature_vectors = []
train_labels = []

# Create plots for training data and collect data
for label, folder in enumerate(train_folders, 1):
    folder_path = os.path.join(train_data_folder, folder)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        coordinates = read_coordinates_from_file(file_path)
        
        if coordinates:
            train_feature_vector = get_feature_vector(coordinates)
            train_feature_vectors.append(train_feature_vector)
            train_labels.append(label)
            plot_coordinates(coordinates, f"Trainingsdaten - {folder} - {file_name}", folder, file_name)

# Create plot for all feature data
plot_3d_data(train_feature_vectors, train_labels)

# convert to numpy-arrays
train_data = np.array(train_feature_vectors)
train_labels = np.array(train_labels)

# Using k-nearest neighbor classification algorithm
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(train_data, train_labels)

# Prediction of activity for test data
test_data_folder = 'data/test'
for i in range(1, 11):
    test_file = os.path.join(test_data_folder, f'{i:02d}.txt')
    test_coordinates = read_coordinates_from_file(test_file)
    test_feature_vector = get_feature_vector(test_coordinates)

    predicted_label = knn_classifier.predict([test_feature_vector])

    activities = {1: 'walking', 2: 'sitting', 3: 'jogging'}
    predicted_activity = activities[predicted_label[0]]

    print(f"prediction for {i}.txt:", predicted_activity)
