import pandas as pd
from imblearn.combine import SMOTEENN
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.read_csv("data_transformed_updated.csv")

X = df.drop("evasão", axis=1)
y = df["evasão"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

imputer = SimpleImputer(strategy="most_frequent")
X = imputer.fit_transform(X)

smoteenn = SMOTEENN(random_state=0, sampling_strategy="auto")
X_resampled, y_resampled = smoteenn.fit_resample(X, y_encoded)

scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.3, random_state=10)

param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

mlp = MLPClassifier(max_iter=100)

grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=3)

grid_search.fit(X_train, y_train)

print("Melhores hiperparâmetros encontrados: ", grid_search.best_params_)

y_probs = grid_search.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0, 1, 200)
f1_scores = [f1_score(y_test, [1 if prob > thr else 0 for prob in y_probs]) for thr in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Melhor Threshold: {best_threshold}")

y_test_pred = [1 if prob > best_threshold else 0 for prob in y_probs]

accuracy = accuracy_score(y_test, y_test_pred) * 100
precision = precision_score(y_test, y_test_pred) * 100
recall = recall_score(y_test, y_test_pred) * 100
f1 = f1_score(y_test, y_test_pred) * 100
roc_auc = roc_auc_score(y_test, y_probs) * 100

print("Accuracy: {:.2f}%".format(accuracy))
print("Precision: {:.2f}%".format(precision))
print("Recall: {:.2f}%".format(recall))
print("F1-Score: {:.2f}%".format(f1))
print("ROC AUC: {:.2f}%".format(roc_auc))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
