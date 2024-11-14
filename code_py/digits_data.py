import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import random
import os

IMAGE_DIR = 'static/digits_images'  # Dossier où sont stockées les images

def get_all_digit_samples():
    digits = load_digits()
    images = digits.images  # Les images 8x8 en nuances de gris
    labels = digits.target  # Les labels (chiffres 0-9)

    # Créer un dictionnaire pour stocker les noms de fichiers et les labels
    image_files = []

    for label in range(10):  # Pour chaque chiffre de 0 à 9
        # Trouver un index correspondant au label
        indices = [i for i, x in enumerate(labels) if x == label]
        if indices:
            random_index = random.choice(indices)
            sample_image = images[random_index]
            image_filename = f"digit_{label}_sample.png"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            
            # Sauvegarder l'image
            plt.imsave(image_path, sample_image, cmap='gray')

            # Ajouter les informations au dictionnaire
            image_files.append({
                'label': label,
                'filename': image_filename
            })

    return image_files
