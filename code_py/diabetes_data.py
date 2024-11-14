import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import random

def get_random_diabetes_sample():
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    data['target'] = diabetes.target
    
    # Choisir un échantillon aléatoire
    random_index = random.choice(data.index)
    sample = data.loc[random_index]
    
    # Extraire les détails
    details = {
        'age': sample['age'],
        'sex': sample['sex'],
        'bmi': sample['bmi'],
        'bp': sample['bp'],
        's1': sample['s1'],
        's2': sample['s2'],
        's3': sample['s3'],
        's4': sample['s4'],
        's5': sample['s5'],
        's6': sample['s6'],
        'target': sample['target']
    }
    
    return details
