# BIJ0013 - task_1
#
# Cílem bylo vytvořit perceptron, který pro 100 generovaných bodů rozhodne, zdali leží nad zadanou přímkou (y = 3x +2), pod ní nebo na ní
# Workflow:
#           Generují se data
#               - 100 XY bodů od [-dimension do dimension] + počítají se reálné "pozice" jednotlivých bodů (tzn. nad/pod/na přimce)
#
#           Trénuje se perceptron
#               - vypočítává se chyba a na základě ní se aktualizují váhy a bios
#
#           Testuje se perceptron
#               - získáváme predikce od natrénovaného perceptronu pro dané body
#
#           Výsledky testování se vykreslí v grafu
#
# Tento perceptron jsem trénoval ve třech "nastaveních" - 2 iterace, 10 iterací a 100 iterací
#   -> Z jednotlivých grafů můžeme vidět, že perceptron natrénovaný pouze ve 2 iteracích není dostačující, některé body neklasifikuje správně.
#      Mezi 10 a 100 iteracemi není žádný viditelný rozdíl (výkon perceptronu je vesměs totožný).
#
#
# Note: Žlutá barva by měla označit body na přímce, nicméně pro daný np.seed se žádný bod nevygeneroval přesně na ní -> žádný žlutý bod není


import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, dimension, n_of_iterations):
        self.points = None
        self.weights = None
        self.bias = None
        self.learning_rate = 0.01
        self.dimension = dimension
        self.n_of_iterations = n_of_iterations

        # -1 = pod přímkou, 0 = na přímce, 1 = nad přímkou
        self.positions = []

        self.predicted_positions = []

    def generate_data(self):
        np.random.seed(111)
        # Generujeme 100 bodů s X a Y souřadnicemi od [-dimension; dimension]
        self.points = np.random.uniform(-self.dimension, self.dimension, (100, 2))

        # Dvě váhy == pro X a Y
        self.weights = np.random.rand(2)

        self.bias = np.random.rand(1)

        # Vypočítáme zda-li bod leží nad přímkou, pod přímkou, na přímce
        #   (tato vypočítaná pozice je použita pouze na oveření efektivity při trénování perceptronu [díky ní víme, zdali perceptron predikoval chybně nebo ne])
        for point in self.points:
            # Vzorec y = 3x + 2
            if point[1] > 3 * point[0] + 2:
                self.positions.append(1)
            elif point[1] < 3 * point[0] + 2:
                self.positions.append(-1)
            else:
                self.positions.append(0)
    def train_perceptron(self):
        for _ in range(self.n_of_iterations):
            index = 0
            # Procházíme všechny body a predikujeme jejich pozici(zda-li bod leží nad přímkou, pod přímkou, na přímce)
            # Pokud predikujeme chybně, aktualizují se váhy a bias
            for point in self.points:
                # Predikujeme třídu(pozici) bodu
                prediction = self.predict(point)

                # Výpočet chyby (rozdíl mezi vypočítanou pozicí a predikovanou pozicí)
                error = self.positions[index] - prediction

                # Aktualizace vah a biasu (pokud je chyba nulová [vypočítaná pozice == predikovaná pozice] ==> nic se neaktualizuje)
                self.weights += self.learning_rate * error * point
                self.bias += self.learning_rate * error

                index += 1
    def predict(self, point):
        # Výpočet váženého součtu
        weighted_sum = np.dot(point, self.weights)
        prediction = weighted_sum + self.bias

        # Proměnnou prediction převedeme do intervalu [-1 ; 1]
        if prediction > 0:
            prediction = 1
        elif prediction < 0:
            prediction = -1
        else:
            prediction = 0

        return prediction

    def test_perceptron(self):
        # Predikujeme třídu(pozici) na natrénovaném perceptronu
        for point in self.points:
            self.predicted_positions.append(self.predict(point))

    def run(self):
        # Generujeme 100 bodů v 2D prostoru od -dimension do dimension
        self.generate_data()

        # Trénujeme perceptron (parametr n_of_iterations)
        self.train_perceptron()

        # Testujeme výkon perceptronu => ukládáme si predikované třídy
        self.test_perceptron()

        # Vykreslení grafu
        self.print_graph()

    def print_graph(self):
        self.predicted_positions = np.array(self.predicted_positions)

        plt.figure(figsize=(12, 8))
        plt.scatter(self.points[self.predicted_positions == 1, 0], self.points[self.predicted_positions == 1, 1], color='red', label='Nad přímkou')
        plt.scatter(self.points[self.predicted_positions == -1, 0], self.points[self.predicted_positions == -1, 1], color='blue', label='Pod přímkou')
        plt.scatter(self.points[self.predicted_positions == 0, 0], self.points[self.predicted_positions == 0, 1], color='yellow', label='Na přímce')

        x_vals = np.linspace(-self.dimension, self.dimension, 100)
        y_vals = -(self.weights[0] * x_vals + self.bias) / self.weights[1]

        plt.plot(x_vals, y_vals, 'm-.', label='Perceptron hranice')
        plt.plot(x_vals, 3 * x_vals + 2, 'k-', label='Přímka y = 3x+2')
        plt.xlim(-self.dimension, self.dimension)
        plt.ylim(-self.dimension, self.dimension)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Perceptron klasifikace při {self.n_of_iterations} iteracích')
        plt.legend()
        plt.grid(True)
        plt.show()
