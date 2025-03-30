# BIJ0013 - task_3
# Cílem bylo vytvořit Hopfieldovu síť pro ukládání a následný "restore" nekompletních nebo jinak pozměněných vzorů
#
# - Workflow:
#     - Vytvoření dat
#         - inicializovala se váhová matice s rozměry size x size (20 x 20) - zde bylo důležité, aby byla symetrická
#     - Trénuje se síť
#         - po přidání nějakých vzorů do pole original_patterns pomocí metody add_pattern_to_memory následuje trénování
#         - každý vzor se postupně převede na 1D pole pomocí flatten (je to z toho důvodu, že jako vstupní vzor ukládáme 2D pole)
#         - následně po reshape získáme sloupcový vektor hodnot
#         - pro výpočet skalárního součinu potřebujeme jak sloupcový, tak řádkový vektor, tzn na daný sloupcový vektor zavoláme .T (neboli transpose)
#         - následně se každý skalární součin přičtě do váhové matice
#         - po iteraci přes všechny vzory jsem se také rozhodl přidat normalizaci vah - mělo by to pomoct stabilitě učení (nejspíše by se to ale projevilo při komplexnější síti, resp. při násobně více vzorech)
#         - následovalo přidání 0 na diagonálu váhové matice
#     - Získávají se opravené vzory
#         - po přidání nějakých vzorů do pole corrupted_patterns pomocí metody add_corrupted_pattern následují "opravy"
#         - zde je podobná myšlena jako u druhého bodu v předchozím kroku - tzn. převedeme 2D pole pomocí flatten na 1D
#         - následuje volba, zdali použijeme synchronní nebo asynchronní přístup
#             - synchronní:
#                 - synchronní přístup má výhodu v rychlosti, nicméně se u něj mohou objevovat artefakty (ghosting) - v mém případě jsem na toto nenarazil
#                 - funguje tak, že se vypočítá skalární součin celé váhové matice a daného "corrupted" vzoru, následně se zavolá na výsledek np.sign(), což vrací hodnoty dle toho, zdali jsou výsledné čísla pozitivní nebo negativní
#             - asynchronní
#                 - asynchronní přístup by měl být více robustnější a odolný vůči problémům synchronního, taky trvá déle
#                 - jeho logika je velmi podobná synchronnímu, nicméně se aktualizuje pouze jeden náhodně vybraný "pixel" v jeden čas
#     - Zobrazení výsledků
#         - prvně jsou zobrazeny vstupní vzory, na kterých se model učil
#         - následně jsou pro každý vstupní vzor zobrazeny jejich "corrupt" varianta a následná opravená varianta
#
# NOTE: Pro učení jsem si vybral 3 vzory na 5x5 mřížce, čísla 5, 7 a písmeno H.
#
# Všechny tyto vzory se mi podařilo úspěšně opravit - respektive jejich "corrupt" varianty.
#
# Problémy nicméně byly, pokud jsem chtěl opravovat např. čísla 5 a 3, jelikož mají podobnou strukturu až na 2 pixely - model v tomto případě predikoval pouze jednu z nich (proto jsem číslo 3 nahradil písmenem H)



import numpy as np
from matplotlib import pyplot as plt


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size

        # Inicializace váhové matice - musí být symetrická
        self.weights = np.zeros((size, size))
        self.original_patterns = []
        self.corrupted_patterns = []
        np.random.seed(123)

    # Pomocná metoda pro naplnění nul na diagonálu
    def diagonal_zeros(self):
        for i in range(self.size):
            self.weights[i][i] = 0

    def add_pattern_to_memory(self, pattern):
        # Nahrazení všech 0 za -1
        pattern[pattern == 0] = -1
        self.original_patterns.append(pattern)

    def add_corrupted_pattern(self, pattern):
        # Nahrazení všech 0 za -1
        pattern[pattern == 0] = -1
        self.corrupted_patterns.append(pattern)

    # Metoda pro trénování "modelu" - respektive jeho váhové matice
    def train_patterns(self):
        for pattern in self.original_patterns:
            # Převedeme pattern na 1D pole - vznikne sloupcový vektor
            pattern = pattern.flatten().reshape((self.size, 1))

            # Vytvoření řádkového vektoru pomocí transpose
            column_pattern = pattern.T

            # Do váhové matice se přičítají jednotlivé výsledky skalárních součinů
            self.weights += np.dot(pattern, column_pattern)

        # Přidání jemné normalizace z důvodu stability učení
        self.weights /= len(self.original_patterns)

        # Přidání nul do diagonály váhové matice
        self.diagonal_zeros()

        self.plot_training()

    # Metoda pro rekonstrukci vzorů
    def recover_patterns(self, synchronous = True):
        for index, corrupted_pattern in enumerate(self.corrupted_patterns):

            # Převedeme pattern na 1D pole - vznikne řádkový
            repaired_pattern = corrupted_pattern.flatten()

            for _ in range(100):
                if synchronous:
                    # Aktualizace všech neuronů v jednom kroku
                    repaired_pattern = np.sign(np.dot(self.weights, repaired_pattern))
                    break
                else:
                    # Postupná aktualizace náhodného neuronu
                    random_index = np.random.randint(0, self.size)
                    repaired_pattern[random_index] = np.sign(np.dot(self.weights[random_index], repaired_pattern))

            self.plot_results(index, corrupted_pattern, repaired_pattern)

    # Vizualizace oprav
    def plot_results(self, index, pattern_corrupt, pattern_recovered):
        plt.subplot(1, 3, 1)
        plt.imshow(self.original_patterns[index].reshape(5, 5), cmap="binary")
        plt.title("Originální vzor")

        plt.subplot(1, 3, 2)
        plt.imshow(pattern_corrupt.reshape(5, 5), cmap="binary")
        plt.title("Poškozený vzor")

        plt.subplot(1, 3, 3)
        plt.imshow(pattern_recovered.reshape(5, 5), cmap="binary")
        plt.title("Obnovený vzor")

        plt.show()

    # Vizualizace trénovaných vzorů
    def plot_training(self):
        num_patterns = len(self.original_patterns)
        for i in range(num_patterns):
            plt.subplot(1, num_patterns, i + 1)
            plt.imshow(self.original_patterns[i].reshape(5, 5), cmap="binary")
            plt.title(f"Pattern {i}")
        plt.show()
