from task_1 import Perceptron
from task_2 import MultiLayerPerceptron


def choose_task():
    print("---------------------")
    print("1. Perceptron: point on the line: 3 body.\n2. Simple neural network: XOR problem: 4 body.\n3. Hopfield network: 4 body.\n4. Q-learning and the game Find the cheese: 4 body.\n5. (Dobrovolné) Pole-balancing problem: 4 body.\n6. L-systems: 3 body.\n7. IFS: 4 body.\n8. TEA - Mandelbrot set or Julia's set: 4 body.\n9. Generation of 2D country using fractal geometry: 4 body.\n10. Theory of chaos: Logistic map, chaotic numbers and their prediction: 4 body.\n11. (Dobrovolné) Chaotic motion - double pendulum: 4 body.\n12. Cellular automata - forest fire algorithm: 3 body.")
    print("---------------------")
    choice = input("Type \"q\" or Choose a task (number): ")
    while choice != "q":
        match choice:
            case "1":
                run_task_1()
            case "2":
                run_task_2()
            case _:
                print("Invalid choice")
        choice = input("Type \"q\" to quit or Choose a task (number): ")


def run_task_1():
    perc = Perceptron(5, 2)
    perc.run()

    perc = Perceptron(5, 10)
    perc.run()

    perc = Perceptron(5, 100)
    perc.run()

def run_task_2():
    mlp = MultiLayerPerceptron(10)
    mlp.run()

    mlp = MultiLayerPerceptron(100)
    mlp.run()

    mlp = MultiLayerPerceptron(1000)
    mlp.run()


if __name__ == "__main__":
    choose_task()