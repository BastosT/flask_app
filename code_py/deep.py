from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data
digits = load_digits()
print(digits.keys())

# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(
    digits.data, 
    digits.target,
    test_size = 0.1
)

# Define model
model = RandomForestClassifier(n_estimators=10)

# Fit Model
model.fit(Xtrain, ytrain)

# Predict model
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred, ytest))