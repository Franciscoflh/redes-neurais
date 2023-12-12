import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import pyplot as plt

# Carregando o dataset Wine
wine = load_wine()
X, y = wine.data, wine.target

# Normalizando os dados
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Dividindo em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)


# Definindo a classe RBFNet
class RBFNet:
    def __init__(self, k, sigma=1.0):
        self.k = k  # número de centros RBF
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def rbf(self, x, c, s):
        return np.exp(-1 / (2 * s ** 2) * cdist(x, c, 'sqeuclidean'))

    def kmeans_centers(self, X, k):
        kmeans = KMeans(n_clusters=k).fit(X)
        return kmeans.cluster_centers_

    def fit(self, X, y):
        self.centers = self.kmeans_centers(X, self.k)
        RBF_X = self.rbf(X, self.centers, self.sigma)
        self.weights = np.linalg.pinv(RBF_X).dot(y)

    def predict(self, X):
        RBF_X = self.rbf(X, self.centers, self.sigma)
        predictions = RBF_X.dot(self.weights)
        return predictions


# Testando diferentes valores de k e sigma
k_values = [5, 10, 15, 20, 25]
sigma_values = [0.5, 1.0, 2.0, 3.0, 4.0]

best_accuracy = 0
best_k, best_sigma = 0, 0

for k in k_values:
    for sigma in sigma_values:
        rbf_net = RBFNet(k=k, sigma=sigma)
        rbf_net.fit(X_train, y_train)
        y_pred = rbf_net.predict(X_test)
        y_pred_classes = np.round(y_pred).astype(int)
        accuracy = accuracy_score(y_test, y_pred_classes)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k, best_sigma = k, sigma

        print(f'k: {k}, sigma: {sigma}, Accuracy: {accuracy}')

print(f'Best Accuracy: {best_accuracy} with k: {best_k}, sigma: {best_sigma}')

# Visualizando a matriz de confusão para a melhor configuração
rbf_net = RBFNet(k=best_k, sigma=best_sigma)
rbf_net.fit(X_train, y_train)
y_pred = rbf_net.predict(X_test)
y_pred_classes = np.round(y_pred).astype(int)
cm = confusion_matrix(y_test, y_pred_classes)


# Função para visualizar a matriz de confusão
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


plot_confusion_matrix(cm, classes=wine.target_names)
