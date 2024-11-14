import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import random
import os

IMAGE_DIR = 'static/iris_images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def plot_iris_sample(sample, filename):
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    values = [sample[feature] for feature in features]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(features, values, color=['blue', 'green', 'red', 'purple'])
    ax.set_xlabel('Colonne')
    ax.set_ylabel('Valeur')
    ax.set_title(f'Iris : {sample["species"]}')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def get_random_iris_sample():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = iris.target_names[iris.target]
    
    random_index = random.choice(data.index)
    sample = data.loc[random_index]
    
    label = sample['species']
    
    image_filename = f"iris_images/{label.lower()}_sample.png"
    image_path = os.path.join(IMAGE_DIR, f"{label.lower()}_sample.png")
    
    plot_iris_sample(sample, image_path)
    
    details = {
        'sepal_length': sample['sepal length (cm)'],
        'sepal_width': sample['sepal width (cm)'],
        'petal_length': sample['petal length (cm)'],
        'petal_width': sample['petal width (cm)'],
        'species': label
    }
    
    return details, image_filename
