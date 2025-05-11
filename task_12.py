from random import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter


class ForestFire():
    def __init__(self):
        self.size = 101
        self.probability_of_tree=0.6
        self.grid = np.random.choice([0, 1], size=(self.size, self.size), p=[1 - self.probability_of_tree, self.probability_of_tree])

        # Pravděpodobnost, že hořící strom nebo prázdné místo bude nahrazeno za živý strom
        self.p = 0.05

        # Pravděpodobnost, že strom sám od sebe začne hořet
        self.f = 0.00001

        # Defaultně zapálíme pravý horní roh
        self.grid[0][self.size-1] = 2

    # Pomocná metoda pro zjištění, zdali soused aktuálního prvku "hoří"
    def neighborOnFire(self, x, y, current_state):
        neighbor_is_on_fire = False
        for i in range(-1,2):
            for j in range(-1,2):
                if 0 <= i+x < self.size and 0 <= j+y < self.size and current_state[i+x][j+y] == 2:
                    neighbor_is_on_fire = True
                    break
            if neighbor_is_on_fire:
                break

        return neighbor_is_on_fire

    def fire(self, current_state):
        # 0 Reprezenture prázdné místo
        # 1 Reprezentuje strom
        # 2 Reprezentuje hořící strom
        new_state = current_state.copy()

        for x in range(self.size):
            for y in range(self.size):
                # Pokud je strom
                if current_state[x][y] == 1:
                    # Pokud jeho soused hoří -> taky hoří
                    if self.neighborOnFire(x,y, current_state):
                        new_state[x][y] = 2
                    #Pokud žádný soused nehoří, tak s pravděpodobností f začne hořet
                    else:
                        new_state[x][y] = 2 if random() < self.f else 1
                # Pokud je prázdná plocha nebo hoří
                elif current_state[x][y] == 0 or current_state[x][y] == 2:
                    new_state[x][y] = 1 if random() < self.p else 0

        return new_state

    def run_simulation(self):
        cmap = colors.ListedColormap(['white', 'green', 'red'])
        fig, ax = plt.subplots()
        img = ax.imshow(self.grid, cmap=cmap)
        original = self.grid.copy()

        def init():
            img.set_data(self.grid)
            ax.set_title("Generace: 0")
            return [img]

        def update(frame):
            self.grid = self.fire(self.grid)
            img.set_data(self.grid)
            ax.set_title(f"Generace: {frame}")

            # Pokud jsme na posledním snímku, ukonči animaci a zavři okno
            if frame == 100:
                plt.close(fig)

            return [img]

        self.ani = FuncAnimation(
            fig,
            update,
            frames=range(1, 101),
            init_func=init,
            blit=False,
            interval=50
        )

        plt.show()

        # Ulož do souboru
        self.grid = original.copy()
        self.ani.save("simulation.gif", writer=PillowWriter(fps=10))
        plt.close()

