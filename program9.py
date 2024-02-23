import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('/content/drive/MyDrive/Iris.csv')

X = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = dataset['Species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

classifier = LogisticRegression()

cv_scores = cross_val_score(classifier, X_scaled, y, cv=5)

mean_accuracy = cv_scores.mean()
print("Mean Accuracy:", mean_accuracy)
