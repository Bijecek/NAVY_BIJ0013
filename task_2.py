# BIJ0013 - task_2
#
# Cílem bylo vytvořit MLP pro XOR problém, který by úspěšně klasifikoval výsledek dvou stupních operací (0/1)
# Workflow:
#           Vytvoření dat
#               - tabulka ze zadání se transformuje do proměnných data_x a data_y (tzn. proměnná pro vstup a proměnná pro očekávaný výstup)
#               - také se inicializují váhy a biasy mezi jednotlivými vrstvami (vstup -> hidden | hidden -> výstup)
#
#           Trénuje se perceptron
#               - vypočítává se chyba, která se počítá jako rozdíl mezi očekávanám výstupem a výsledkem z vrstev (vstup -> hidden | hidden -> výstup),
#               k tomu se využívá funkce sigmoid, která normalizuje výsledek do intervalu [0;1]
#               - následně se provádí "backpropagace" neboli zpětný průchod sítí -> získáváme gradient ( k tomu pomáhá funkce calc_dsigmoid )
#               - pomocí získaných gradientů následně aktualizujeme váhy a biasy
#
#           Testuje se perceptron
#               - získáváme predikce od natrénovaného perceptronu pro dané vstupy ( výsledek je hodnota v intervalu 0-1)
#
#           Zobrazení výsledků
#               - výsledky jsou zobrazeny v intervalu 0-1 pro každý vstup (velkem jsou 4 vstupy)
#
# Tento MLP jsem trénoval ve třech "nastaveních" - 10 iterací, 100 iterací a 1000 iterací
#   -> Z jednotlivých výsledků můžeme vidět, že MLP při 10 a 100 iterací stále nedokáže úspěšně klasifikovat daný XOR problém.
#   -> Při 1000 iteracích jsou již predikce stabilní:
                                                    # [[0.0626129 ]
                                                    #  [0.93033701]
                                                    #  [0.92729364]
                                                    #  [0.08738947]]
#
#
# NOTE: V úvahu se zde musí brát můj learning_rate=0.5 a počet neuronů ve skryté vrstvě (hidden_size = 8), pro jiné nastavení by i výsledky mohly být jiné



import numpy as np


class MultiLayerPerceptron:
    def __init__(self, n_of_iterations):
        self.data_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.data_y = np.array([[0], [1], [1], [0]])
        self.learning_rate = 0.5
        self.n_of_iterations = n_of_iterations

        # Počet vstupů - vždy máme pole dvou čísel ==> 2
        self.input_size = 2
        # Počet skrytých neuronů
        self.hidden_size = 8
        # Počet výstupů - číslo 0 nebo 1
        self.output_size = 1

        # Inicializace defaulních vah
        # Mezi vstupní a skrytou vrstvou
        self.input_hidden_weights = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        # Mezi skrytou a výstupní vrstvou
        self.hidden_output_weights = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))

        self.hidden_bias = np.random.uniform(-1, 1, (1, self.hidden_size))
        self.output_bias = np.random.uniform(-1, 1, (1, self.output_size))

    # Funkce pro vraceni hodnoty mezi 0 az 1
    def calc_sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    # Derivace sigmoid funkce - pouzivame u backpropagace
    def calc_dsigmoid(self, value):
        return value * (1 - value)

    def train_mlp(self):
        for i in range(self.n_of_iterations):
            input_hidden_res, hidden_output_res = self.predict_mlp(self.data_x)

            # Výpočet chyby (rozdíl mezi "reálným" a predikovaným výsledkem)
            error = self.data_y - hidden_output_res

            # Backpropagace chyby - tzn počítáme gradient
            gradient_output = error * self.calc_dsigmoid(hidden_output_res)
            gradient_hidden = np.dot(gradient_output, self.hidden_output_weights.T) * self.calc_dsigmoid(input_hidden_res)

            # Aktualizujeme váhy a biasy
            self.input_hidden_weights += np.dot(self.data_x.T, gradient_hidden) * self.learning_rate
            self.hidden_output_weights += np.dot(input_hidden_res.T, gradient_output) * self.learning_rate

            self.output_bias += np.sum(gradient_output, axis=0, keepdims=True) * self.learning_rate
            self.hidden_bias += np.sum(gradient_hidden, axis=0, keepdims=True) * self.learning_rate
    def predict_mlp(self, data):
        # Počítání váženého součtu mezi vstupní a skrytou vrstvou
        input_hidden = np.dot(data, self.input_hidden_weights) + self.hidden_bias
        input_hidden_res = self.calc_sigmoid(input_hidden)

        # Počítání váženého součtu mezi skrytou a výstupní vrstvou
        hidden_output = np.dot(input_hidden_res, self.hidden_output_weights) + self.output_bias
        hidden_output_res = self.calc_sigmoid(hidden_output)

        return input_hidden_res, hidden_output_res

    def run(self):
        self.train_mlp()
        _, output = self.predict_mlp(self.data_x)
        print("--------------------")
        print(f"Výsledky predikce pro XOR problém, MLP trénované {self.n_of_iterations} iterací:")
        print(output)
        print("--------------------")


