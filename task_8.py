import numpy as np
import matplotlib.pyplot as plt

class TEA:
    def __init__(self, constant, n_of_iter):
        self.constant = constant

        # Inicializace defaulních hodnot pro Julia set
        # Reálná část
        self.real_num_range = (-1.5, 1.5)
        # Imaginární část
        self.imaginary_num_range = (-1.5, 1.5)
        self.size = 650

        self.n_of_iter = n_of_iter
        self.results = None

    def compute_julia_set(self):
        # Vytvoření matic pro reálný a imaginární space
        real_space = np.linspace(self.real_num_range[0], self.real_num_range[1], self.size)
        imaginary_space = np.linspace(self.imaginary_num_range[0], self.imaginary_num_range[1], self.size)

        # Vytvoření mřížek
        real_grid, imaginary_grid = np.meshgrid(real_space, imaginary_space)

        # Vytvoření komplexních čísel
        complex_numbers = real_grid + 1j * imaginary_grid

        # Inicializace výsledné matice
        self.results = np.zeros(complex_numbers.shape, dtype=int)

        for i in range(self.n_of_iter):
            # Vytvoříme si masku, která ověří podmínku, že komplexní číslo je <= 2
            valid_numbers = np.abs(complex_numbers) <= 2

            # Proběhne aktualizace výsledkové matice na daných pozicích čísel, které splnila podmínku
            self.results[valid_numbers] = i

            # Aktualizují se komplexní čísla (ty, které splnila podmínku)
            complex_numbers[valid_numbers] = complex_numbers[valid_numbers] ** 2 + self.constant

    def print_results(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.results, cmap='inferno')
        plt.title(f"Julia set with constant {self.constant}, {self.n_of_iter} iterations")
        plt.show()
