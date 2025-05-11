import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.src import layers


class TheoryOfChaos:
    def __init__(self, start, stop, num):
        # Inicializace defaultních parametrů
        self.a_values = np.linspace(start, stop, num)
        self.iterations = 1000
        self.last = 200
        self.x = 1e-5 * np.ones(len(self.a_values))

        self.results = []
        self.X_train = []
        self.y_train = []

        self.model = None

        self.a_plot=[]

    # Implementace logistické mapy
    def compute(self):
        for i in range(self.iterations):
            self.x = self.a_values * self.x * (1-self.x)
            if i >= (self.iterations - self.last):
                self.results.append((self.a_values.copy(), self.x.copy()))

    # Zobrazení Bifurkačního grafu
    def print_results(self):
        for a_value, x in self.results:
            plt.plot(a_value, x, ',k', alpha=0.4)
            self.a_plot.append(a_value)

        plt.title("Bifurkační diagram")
        plt.xlabel("a_value")
        plt.ylabel("x")
        plt.show()

    # Trénování modelu pomocí Kerasu
    def train_model(self):
        # Vytváření trénovacích dat
        a_train = np.random.uniform(self.a_values[0], self.a_values[-1], 10000)
        x_train = np.random.uniform(0, 1, 10000)
        # Ukládání dalšího kroku
        y_train = a_train * x_train * (1 - x_train)

        X_train = np.column_stack((a_train, x_train))
        y_train = y_train.reshape(-1, 1)

        model = keras.models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(2,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=5, batch_size=32)

        self.model = model
        return model

    # Zobrazení predikce Bifurkačního grafu
    def plot_results(self):
        x_pred = 1e-5 * np.ones(len(self.a_values))
        x_pred_plot = []

        for i in range(self.iterations):
            input_pred = np.column_stack((self.a_values, x_pred))
            x_pred = self.model.predict(input_pred, verbose=0).flatten()
            x_pred = np.clip(x_pred, 0, 1)
            if i >= (self.iterations - self.last):
                x_pred_plot.append(x_pred.copy())

        x_pred_plot = np.array(x_pred_plot).flatten()
        a_plot = np.array(self.a_plot).flatten()

        plt.figure(figsize=(10, 6))
        plt.plot(a_plot, x_pred_plot, ',r', alpha=0.25)
        plt.title("Predikovaný Bifurkační Diagram pomocí NN")
        plt.xlabel("a_value")
        plt.ylabel("x")
        plt.show()
