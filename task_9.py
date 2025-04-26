import random

from matplotlib import pyplot as plt


class CountryGeneration:
    def __init__(self, init_start_pos, init_end_pos, y_offset, n_of_iterations, color):
        self.start_pos = init_start_pos
        self.end_pos = init_end_pos

        self.y_offset = y_offset
        self.n_of_iterations = n_of_iterations

        plt.figure(figsize=(10, 6))

        self.color = color

    # Pomocná funkce pro více scén
    def change_settings(self, init_start_pos, init_end_pos, y_offset, n_of_iterations, color):
        self.start_pos = init_start_pos
        self.end_pos = init_end_pos

        self.y_offset = y_offset
        self.n_of_iterations = n_of_iterations

        self.color = color

    def generate_landscape(self):
        lines = [(self.start_pos[0], self.start_pos[1], self.end_pos[0], self.end_pos[1])]

        x_values = []
        y_values = []

        for index in range(self.n_of_iterations):
            new_lines = []

            for index_line, (x_start, y_start, x_end, y_end) in enumerate(lines):
                # Najdeme prostředek úsečky
                middle_x = (x_start + x_end) / 2
                middle_y = (y_start + y_end) / 2

                # Rozhodnutí, zdali budeme tvořit pod nebo nad aktuální úsečku
                offset = random.uniform(-self.y_offset, self.y_offset)

                # Hodnotota Y se tedy o tento offset posune
                middle_y = middle_y + offset

                # Vytvoří se tedy dvě nové úsečky: start -> NEW a NEW -> end, tyto úsečky se uloží
                new_lines.append((x_start, y_start, middle_x, middle_y))
                new_lines.append((middle_x, middle_y, x_end, y_end))

                # Vykreslujeme pouze v poslední iteraci
                if index == (self.n_of_iterations - 1):
                    # Můžeme vykreslovat jednotlivé úsečky, nicméně to při více scénách vytváří překrývací problémy
                    #plt.plot([x_start, middle_x], [y_start, middle_y], color=self.color)
                    #plt.plot([middle_x, x_end], [middle_y, y_end], color=self.color)

                    # Uložíme si X a Y hodnoty pro následné vykreslení prostoru pod křivkou
                    x_values.extend([x_start, middle_x])
                    y_values.extend([y_start, middle_y])

                    # Pokud máme poslední line, vezmeme i její konec
                    if index_line == len(lines) - 1:
                        x_values.extend([middle_x, x_end])
                        y_values.extend([middle_y, y_end])


            # Další iteraci se bude generovat z již přetvořených úseček
            lines = new_lines

        # Vykreslíme prostor pod křivkou
        plt.fill_between(x_values, y_values, -500, color=self.color)

    def show_result(self):
        plt.ylim(-500, 500)
        plt.axis('off')

        plt.show()