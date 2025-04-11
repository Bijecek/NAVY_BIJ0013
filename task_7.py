import random

from matplotlib import pyplot as plt

class IFS:
    def __init__(self, model_no):
        self.model_no = model_no
        self.transformations = None

        # Pole pro historie
        self.x_history = []
        self.y_history = []
        self.z_history = []
        self.choose_model()

    # Metoda pro definici defaultních transformací a výběru aktuálního modelu
    def choose_model(self):
        transformations1 = [
            [0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
            [0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00],
            [-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00],
            [0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00]
        ]

        transformations2 = [
            [0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
            [0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45, 0.00, 1.00, 0.00],
            [-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45, 0.00, 1.25, 0.00],
            [0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49, 0.00, 2.00, 0.00]
        ]


        match self.model_no:
            case 1:
                self.transformations = transformations1
            case 2:
                self.transformations = transformations2

    def compute_transformations(self):
        x, y, z = 0, 0, 0

        # Počet iterací
        for _ in range(100000):
            # Procházíme každý řádek v modelu
            for row in self.transformations:
                random_val = random.random()

                # Pokaždé máme 25% šanci, že se provede daný výpočet
                if random_val < 0.25:
                    a, b, c, d, e, f, g, h, i, j, k, l = row

                    # Násobení a sčítání jednotlivých matic
                    x = a*x + b*y + c*z + j
                    y = d*x + e*y + f*z + k
                    z = g*x + h*y + i*z + l

                    # Přidání do historie pro vizualizaci
                    self.x_history.append(x)
                    self.y_history.append(y)
                    self.z_history.append(z)


    def visualize(self):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_history, self.y_history, self.z_history, s=5.0, color="black")

        plt.show()