import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

cc_data = pd.read_csv('cc_approvals_downloadable.data', header=None)

# Tratando valores faltantes
cc_data = cc_data.replace("?", np.nan)
for col in cc_data.columns:
    if cc_data[col].dtype == 'object':
        cc_data[col] = cc_data[col].fillna(cc_data[col].mode().iloc[0])
    else:
        cc_data[col] = cc_data[col].fillna(cc_data[col].mean())

# Convertendo dados categóricos para numéricos
le = LabelEncoder()
for col in cc_data.columns:
    if cc_data[col].dtype == 'object':
        cc_data[col] = le.fit_transform(cc_data[col])

# Dividindo os dados
X = cc_data.drop(15, axis=1)
y = cc_data[15]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Implementação Adaline
class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


# Treinando o modelo Adaline
adaline = AdalineGD(eta=0.001, n_iter=100)
adaline.fit(X_train, y_train)

# Previsões e avaliação
y_pred = adaline.predict(X_test)
y_pred_class = np.where(y_pred > 0.5, 1, 0)

# Calculando métricas adicionais
accuracy = (y_pred_class == y_test).mean()
conf_matrix = confusion_matrix(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)
