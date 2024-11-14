from flask import Flask, render_template, flash, flash, request, jsonify
from code_py.temp import calculer_indice_de_chaleur
from code_py.iris_data import get_random_iris_sample
from code_py.diabetes_data import get_random_diabetes_sample
from code_py.digits_data import get_all_digit_samples
from sqlalchemy.orm import joinedload
from sqlalchemy import text
import os
from Models.models import db  
from Models.models import * 
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_diabetes, load_digits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.datasets import cifar100 , mnist
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D , Activation
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import PolynomialFeatures


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, 'database/chinook.db')
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False 

db.init_app(app)


# Configuration InfluxDB
app.config['INFLUXDB_V2_URL'] = "http://192.168.168.88:8086"
app.config['INFLUXDB_V2_ORG'] = "IUT"
app.config['INFLUXDB_V2_TOKEN'] = 'gQMgdAee398j6l9G6f5LcDNXwNJONYcPX5MDP-oZlX8oiVfkojD7nl-SnmA9BHI2drpgIAlWIjf-3e1WPejcqw=='

# Instanciation de la classe InfluxDB
client = InfluxDBClient(
    url=app.config['INFLUXDB_V2_URL'],
    token=app.config['INFLUXDB_V2_TOKEN'],
    org=app.config['INFLUXDB_V2_ORG']
)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ia', methods=['GET', 'POST'])
def ia():
    results_description = ''
    results_data = ''

    # Valeurs par défaut des hyperparamètres
    model_name = 'iris'
    test_size = 10  # Valeur par défaut en pourcentage
    n_estimators = 10  # Valeur par défaut

    if request.method == 'POST':
        model_name = request.form['model']
        test_size = float(request.form['test_size'])
        n_estimators = int(request.form['n_estimators'])

        # Charger les données selon le modèle choisi
        if model_name == 'iris':
            dataset = load_iris()
        elif model_name == 'diabetes':
            dataset = load_diabetes()
        else:  # 'digit'
            dataset = load_digits()

        # Split data
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            dataset.data,
            dataset.target,
            test_size=test_size / 100
        )

        # Choisir le modèle selon le type de problème
        if model_name == 'diabetes':
            model = RandomForestRegressor(n_estimators=n_estimators)
            # Fit Model
            model.fit(Xtrain, ytrain)

            # Predict model
            ypred = model.predict(Xtest)

            # Calculer le Mean Absolute Percentage Error (MAPE) pour la régression
            mape = metrics.mean_absolute_percentage_error(ytest, ypred)

            # Mettre à jour la description des résultats
            results_description = f"Résultats du modèle {model_name} avec une taille de test de {test_size}% et {n_estimators} estimateurs."
            results_data = f"Mean Absolute Percentage Error : {mape:.2%}"


        else:
            model = RandomForestClassifier(n_estimators=n_estimators)
            # Fit Model
            model.fit(Xtrain, ytrain)

            # Predict model
            ypred = model.predict(Xtest)

            # Générer le rapport de classification
            report = metrics.classification_report(ytest, ypred)

            # Mettre à jour la description des résultats
            results_description = f"Résultats du modèle {model_name} avec une taille de test de {test_size}% et {n_estimators} estimateurs."
            results_data = report

    return render_template('ia.html', 
                           results_description=results_description, 
                           results_data=results_data,
                           selected_model=model_name,
                           test_size=test_size,
                           n_estimators=n_estimators)


@app.route('/iaten', methods=['GET', 'POST'])
def ia2():
    results_description = ''
    test_loss = None
    test_accuracy = None

    # Valeurs par défaut des hyperparamètres
    test_size = 10  # Par défaut, 10%
    batch_size = 128  # Par défaut
    epochs = 10  # Par défaut
    validation_split = 0.1  # Par défaut

    if request.method == 'POST':
        # Récupérer les valeurs des hyperparamètres
        test_size = float(request.form['test_size'])
        batch_size = int(request.form['batch_size'])
        epochs = int(request.form['epochs'])
        validation_split = float(request.form['validation_split'])

        # Charger le jeu de données MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Remodeler les images
        img_height, img_width = X_train.shape[1:3]
        X_train = X_train.reshape((X_train.shape[0], img_height * img_width))
        X_test = X_test.reshape((X_test.shape[0], img_height * img_width))

        # Encoder les étiquettes
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        # Définir le modèle
        xi = Input(shape=(img_height * img_width,))
        x = Dense(10)(xi)
        y = Activation('softmax')(x)
        model = Model(inputs=[xi], outputs=[y])

        # Compiler le modèle
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Entraîner le modèle
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split / 100)

        # Évaluer le modèle
        score = model.evaluate(X_test, y_test, verbose=0)
        test_loss = round(score[0], 2)  # Arrondir à deux chiffres
        test_accuracy = round(score[1], 2)  # Arrondir à deux chiffres
        

        # Mettre à jour la description des résultats
        results_description = "Résultats de l'entraînement du modèle sur le jeu de données MNIST."

    return render_template('iaten.html',
                           results_description=results_description,
                           test_loss=test_loss,
                           test_accuracy=test_accuracy,
                           test_size=test_size,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_split=validation_split)

@app.route('/train', methods=['GET', 'POST'])
def train():
    # Valeurs par défaut des hyperparamètres
    batch_size = 50
    no_epochs = 5
    validation_split = 0.2
    test_loss = None
    test_accuracy = None
    history = None

    if request.method == 'POST':
        # Récupérer les hyperparamètres du formulaire
        batch_size = int(request.form.get('batch_size', batch_size))
        no_epochs = int(request.form.get('epochs', no_epochs))
        validation_split = float(request.form.get('validation_split', validation_split))

        # Charger les données CIFAR-100
        (input_train, target_train), (input_test, target_test) = cifar100.load_data()

        # Normaliser les données
        input_train = input_train.astype('float32') / 255
        input_test = input_test.astype('float32') / 255

        # Créer le modèle
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(100, activation='softmax'))

        # Compiler le modèle
        model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

        # Entraîner le modèle
        history = model.fit(input_train, target_train, batch_size=batch_size, epochs=no_epochs,
                            validation_split=validation_split / 100, verbose=1)

        # Évaluer le modèle
        score = model.evaluate(input_test, target_test, verbose=0)
        test_loss = score[0]
        test_accuracy = score[1]

        # Sauvegarder l'historique
        history_loss = history.history['val_loss']
        history_accuracy = history.history['val_accuracy']

        # Sauvegarder les graphes
        plt.figure()
        plt.plot(history_loss)
        plt.title('Validation Loss')
        plt.savefig('static/val_loss.png')
        plt.close()

        plt.figure()
        plt.plot(history_accuracy)
        plt.title('Validation Accuracy')
        plt.savefig('static/val_accuracy.png')
        plt.close()

    return render_template('train.html', 
                           test_loss=test_loss, 
                           test_accuracy=test_accuracy,
                           history={'val_loss': 'static/val_loss.png', 'val_accuracy': 'static/val_accuracy.png'},
                           batch_size=batch_size,
                           no_epochs=no_epochs
                           )




@app.route('/bitcoin')
def bitcoin():
    try:
        query_api = client.query_api()
        bucket_name = "Bitcoin"
        query = f'''
        from(bucket: "{bucket_name}")
        |> range(start: -1w)
        '''
        
        # Exécution de la requête InfluxDB
        result = query_api.query(query)
        
        # Stocker les données historiques
        data = []
        for table in result:
            for record in table.records:
                data.append({
                    "time": record.get_time().timestamp(),  # Convertir le temps en timestamp
                    "value": record.get_value()
                })
        
        # Convertir en DataFrame
        timeserie = pd.DataFrame(data)
        
        # Supprimer les doublons basés sur les timestamps
        timeserie = timeserie.drop_duplicates(subset='time', keep='first')
        
        # Convertir la série temporelle en liste de dictionnaires
        timeserie_list = timeserie.to_dict(orient='records')  # Historique
        
        # Rendre le template avec uniquement les données historiques
        return render_template('bitcoin.html', timeserie=timeserie_list)
        
    except Exception as e:
        return f"Erreur lors de la récupération des données : {e}"




@app.route('/temp', methods=['GET', 'POST'])
def temp():
    temperature_ressentie = None
    temperature_ressentie_f = None
    temperature = None
    humidite_air = None

    if request.method == 'POST':
        temperature = request.form.get('temperature')
        humidite_air = request.form.get('humidite_air')

        if temperature and humidite_air:
            try:
                temperature = float(temperature)
                humidite_air = float(humidite_air)
                temperature_ressentie = calculer_indice_de_chaleur(temperature, humidite_air)
                temperature_ressentie_f = (temperature_ressentie * 9 / 5) + 32
            except ValueError:
                flash("Erreur : veuillez entrer des valeurs valides pour la température et l'humidité.", 'danger')
        else:
            flash("Erreur : veuillez remplir tous les champs.", 'danger')

    return render_template(
        'temp.html',
        temperature_ressentie=temperature_ressentie,
        temperature_ressentie_f=temperature_ressentie_f,
        temperature=temperature,
        humidite_air=humidite_air
    )

@app.route('/iris')
def iris():
    details, image_filename = get_random_iris_sample()
    return render_template('iris.html', details=details, image_filename=image_filename)

@app.route('/diabetes')
def diabetes():
    details = get_random_diabetes_sample()
    return render_template('diabetes.html', details=details)

@app.route('/digits')
def digits():
    image_files = get_all_digit_samples()
    return render_template('digits.html', image_files=image_files)

@app.route('/actions')
def actions():
    return render_template('actions.html')

@app.route('/artiste_album')
def artiste_album():
    artists_data = Artist.query.options(joinedload(Artist.albums)).all()

    artists_with_albums = []
    for artist in artists_data:
        albums = [album.Title for album in artist.albums]  # Utiliser la relation 'albums' définie dans le modèle
        artists_with_albums.append({
            'Name': artist.Name,
            'Albums': albums
        })

    return render_template('artiste_album.html', artists=artists_with_albums)

@app.route('/artiste')
def artiste():
    # Récupérer tous les artistes
    artists_data = Artist.query.all()

    # Créer une structure de données pour les artistes
    artists_list = [{'Name': artist.Name} for artist in artists_data]

    return render_template('artiste.html', artists=artists_list)

@app.route('/album')
def album():
    # Récupérer tous les albums
    albums_data = Album.query.all()

    # Créer une structure de données pour les albums
    albums_list = [{'Title': album.Title} for album in albums_data]

    return render_template('album.html', albums=albums_list)

@app.route('/question', methods=['GET'])
def get_answer():
    keyword = request.args.get('keyword')
    response_data = []
    description = 'Aucune réponse à afficher.'

    if keyword == 'liste_artiste':
        artists = Artist.query.all()
        response_data = [artist.to_dict() for artist in artists]
        description = 'Affichage de la liste des artistes.'
    
    elif keyword == 'nb_album_artiste':
        artists_data = Artist.query.options(joinedload(Artist.albums)).all()
        response_data = [{artist.Name: len(artist.albums)} for artist in artists_data]
        description = 'Affichage du nombre d\'album par artiste.'

    elif keyword == 'artistes_populaires':
        # Récupérer les artistes les plus populaires par le nombre de ventes
        popular_artists = db.session.execute(text("""
            SELECT a.Name, COUNT(i.InvoiceId) AS NumSales
            FROM artists a
            JOIN albums al ON a.ArtistId = al.ArtistId
            JOIN tracks t ON al.AlbumId = t.AlbumId
            JOIN invoice_items i ON t.TrackId = i.TrackId
            GROUP BY a.ArtistId
            ORDER BY NumSales DESC
            LIMIT 10
        """)).fetchall()

        response_data = [{'Name': row[0], 'NumSales': row[1]} for row in popular_artists]
        description = 'Affichage des 10 clients les plus populaires.'


    elif keyword == 'clients_depensiers':
        top_spenders = db.session.execute(text("""
            SELECT c.FirstName || ' ' || c.LastName AS CustomerName, ROUND(SUM(i.Total), 2) AS TotalSpent
            FROM customers c
            JOIN invoices i ON c.CustomerId = i.CustomerId
            GROUP BY c.CustomerId
            ORDER BY TotalSpent DESC
            LIMIT 10
        """)).fetchall()

        response_data = [{'CustomerName': row[0], 'TotalSpent': row[1]} for row in top_spenders]
        description = 'Affichage des 10 clients ayant dépensé le plus d\'argent.'

    elif keyword == 'genres_pistes':
        genres_data = db.session.execute(text("""
            SELECT g.Name AS GenreName, COUNT(t.TrackId) AS TrackCount
            FROM genres g
            LEFT JOIN tracks t ON g.GenreId = t.GenreId
            GROUP BY g.GenreId
            ORDER BY TrackCount DESC
        """)).fetchall()

        response_data = [{'GenreName': row[0], 'TrackCount': row[1]} for row in genres_data]
        description = 'Affichage de tous les genres musicaux et le nombre de pistes associées.'

    return render_template('actions.html', response_data=response_data, keyword=keyword, description=description)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)



