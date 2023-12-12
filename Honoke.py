import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Carregando e normalizando o conjunto de dados Iris
iris = load_iris()
data = iris.data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Definição da classe KohonenNet
class KohonenNet:
    def __init__(self, net_size, input_dim):
        self.net_size = net_size
        self.input_dim = input_dim
        self.weights = np.random.random((net_size[0], net_size[1], input_dim))

    def train(self, data, num_epochs, learning_rate, radius):
        self.bmu_counts = np.zeros(self.net_size)
        for epoch in range(num_epochs):
            total_dist = 0
            old_weights = np.copy(self.weights)

            for i in range(len(data)):
                bmu, dist = self.find_bmu(data[i])
                total_dist += dist
                self.bmu_counts[bmu[0], bmu[1]] += 1
                self.update_weights(data[i], bmu, learning_rate, radius)

            avg_dist = total_dist / len(data)
            total_error = np.sqrt(total_dist)
            weight_change = np.sum(np.abs(self.weights - old_weights))

            print(f'Epoch {epoch+1}: Avg Distance: {avg_dist}, Total Error: {total_error}, Weight Change: {weight_change}')

            learning_rate *= 0.95
            radius *= 0.95

    def find_bmu(self, data_point):
        bmu_idx = np.array([0, 0])
        min_dist = np.inf
        for x in range(self.net_size[0]):
            for y in range(self.net_size[1]):
                w = self.weights[x, y, :]
                sq_dist = np.sum((w - data_point) ** 2)
                if sq_dist < min_dist:
                    min_dist = sq_dist
                    bmu_idx = np.array([x, y])
        return bmu_idx, min_dist

    def update_weights(self, data_point, bmu, learning_rate, radius):
        for x in range(self.net_size[0]):
            for y in range(self.net_size[1]):
                w = self.weights[x, y, :]
                dist_to_bmu = np.sum((np.array([x, y]) - bmu) ** 2)
                if dist_to_bmu < radius ** 2:
                    influence = np.exp(-dist_to_bmu / (2 * (radius ** 2)))
                    self.weights[x, y, :] += learning_rate * influence * (data_point - w)

    def plot_bmu_distribution(self):
        plt.imshow(self.bmu_counts, cmap='hot', interpolation='nearest')
        plt.title('BMU Distribution')
        plt.colorbar()
        plt.show()

    def plot_weights(self):
        fig, axes = plt.subplots(self.net_size[0], self.net_size[1], figsize=(8, 8))
        fig.suptitle('Neuron Weights')
        for i in range(self.net_size[0]):
            for j in range(self.net_size[1]):
                axes[i, j].bar(range(self.input_dim), self.weights[i, j, :])
                axes[i, j].set_ylim([0, 1])
                axes[i, j].axis('off')
        plt.show()

# Criando e treinando a rede
net = KohonenNet(net_size=(10, 10), input_dim=4)
net.train(normalized_data, num_epochs=100, learning_rate=0.1, radius=5)

# Após o treinamento, plotar as visualizações
net.plot_bmu_distribution()
net.plot_weights()