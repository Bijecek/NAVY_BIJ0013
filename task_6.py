import math

from matplotlib import pyplot as plt

class LSystem:
    def __init__(self, axiom, rule, angle, num_of_nesting):
        self.axiom = axiom
        self.rule = rule
        self.angle = angle
        self.num_of_nesting = num_of_nesting

        self.instructions = ""
        self.drawing_data = []

    # Metoda, která transformuje axiom a pravidlo na sekvenci instrukcí
    def compute(self):
        current_axiom = self.axiom

        for _ in range(self.num_of_nesting):
            for char in current_axiom:
                match char:
                    case 'F':
                        self.instructions += self.rule
                    case _:
                        self.instructions += char

            current_axiom = self.instructions
            self.instructions = ""

        self.instructions = current_axiom
    # Metoda pro procházení jednotlivých instrukcí a vykonávání příslušných akcí
    def perform_instructions(self):
        pos_x, pos_y = 0.0, 0.0
        direction = 0
        stack = []

        for instruction in self.instructions:
            match instruction:
                case 'F':
                    # Používáme radiány z důvodu konvence
                    rad = math.radians(direction)

                    # Cosinus úhlu počítá posun na ose X
                    new_x = pos_x + math.cos(rad)

                    # Sinus úhlu počítá posun na ose Y
                    new_y = pos_y + math.sin(rad)
                    self.drawing_data.append(((pos_x, pos_y), (new_x, new_y)))

                    # Aktualizujeme pozici
                    pos_x, pos_y = new_x, new_y
                case '+':
                    direction -= self.angle
                case '-':
                    direction += self.angle
                case '[':
                    stack.append((pos_x, pos_y, direction))
                case ']':
                    pos_x, pos_y, direction = stack.pop()

    # Metoda pro vizualizaci
    def visualize(self):
        fig, ax = plt.subplots()
        for (start, end) in self.drawing_data:
            ax.plot([start[0], end[0]], [start[1], end[1]], color="black")

        ax.set_aspect('equal')
        ax.axis('off')
        plt.show()
