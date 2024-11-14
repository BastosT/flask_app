Flask App - README
Description

Cette application Flask utilise l'intelligence artificielle pour l'analyse de données de différents datasets, intégrant des modèles de machine learning. Elle permet de manipuler et d'explorer des ensembles de données populaires tels que IRIS, MNIST, Digits, et Diabetes. L'application s'appuie sur une architecture modulaire, facilitant l'intégration de nouvelles fonctionnalités et l'utilisation de différentes bases de données.
Architecture de l'Application

L'application se compose des modules principaux suivants :
1. Datasets

Cette section regroupe les ensembles de données utilisés pour l'entraînement et la prédiction via des modèles de machine learning. Voici quelques datasets supportés :

    CIFAR100 : Un dataset d'images utilisé pour des tâches de classification d'images complexes.
    MNIST : Dataset standard de chiffres manuscrits pour la reconnaissance de chiffres.
    Digits : Dataset similaire à MNIST mais avec des images plus petites et en nuances de gris.
    Diabetes : Utilisé pour des tâches de prédiction liées au diabète.
    IRIS : Dataset classique pour la classification des espèces de fleurs.

2. IA (Intelligence Artificielle)

Cette partie se concentre sur l'implémentation des modèles de machine learning et deep learning :
Modèles Implémentés

    Random Forest : Utilisé pour la classification de données tabulaires, avec des ajustements d'hyperparamètres pour optimiser la performance.
    Neural Network (Réseau de Neurones) :
        CNN (Convolutional Neural Network) : Utilisé principalement pour les tâches de classification d'images (ex. MNIST, CIFAR100).
        Dense : Réseaux de neurones fully connected, utilisés pour des tâches de prédiction plus simples.

Hyperparamètres

Les modèles peuvent être ajustés via des hyperparamètres tels que le nombre d'arbres pour les Random Forests ou le nombre de couches dans les réseaux de neurones, pour améliorer les performances.
3. Libraries Utilisées (Python)

L'application s'appuie sur un ensemble de bibliothèques Python populaires pour l'analyse de données et l'apprentissage automatique :

    Numpy : Pour les opérations mathématiques et le traitement des matrices.
    Pandas : Pour la manipulation et l'analyse de données structurées.
    Scikit-learn : Pour les modèles d'apprentissage automatique classiques tels que les Random Forests.
    TensorFlow : Pour les réseaux de neurones et l'entraînement de modèles deep learning.
    Matplotlib : Pour la visualisation des données et des résultats des modèles.

4. Base de Données

L'application peut utiliser plusieurs bases de données pour le stockage et la gestion des données :
Bases de Données Supportées :

    SQLite : Utilisée pour les données locales simples et pour le stockage de petites données d'entraînement.
    InfluxDB : Utilisée pour le stockage de séries temporelles, particulièrement utile pour les analyses de données de capteurs ou les logs de performances des modèles.
    Alchemy : Utilisée pour la gestion des opérations SQL avec SQLAlchemy, facilitant les requêtes complexes sur les données.

Tables Modélisées :

    Chinook : Exemple de base de données utilisée pour tester les opérations SQL, incluant des tables comme Artist et Album.

5. Composants du Machine Learning

L'application permet d'entraîner des modèles sur les datasets disponibles et de les évaluer :

    Chargement des Données : Les datasets peuvent être chargés dynamiquement pour entraîner différents modèles.
    Entraînement des Modèles : Les modèles sont entraînés sur les datasets pour obtenir des prédictions et évaluer la performance.
    Évaluation des Modèles : Les performances des modèles sont évaluées à l'aide de métriques comme l'accuracy, le recall, et le F1-score.


Utilisation

    L'application permet d'entraîner des modèles sur différents datasets disponibles via une interface utilisateur.
    Les résultats des modèles, y compris les visualisations et les prédictions, sont affichés directement sur l'interface.

    
