import math

import numpy as np
from matplotlib import pyplot as plt

from task_1 import Perceptron
from task_2 import MultiLayerPerceptron
from task_3 import HopfieldNetwork
from task_4 import QLearning
from task_6 import LSystem
from task_7 import IFS


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
            case "3":
                run_task_3()
            case "4":
                run_task_4()
            case "6":
                run_task_6()
            case "7":
                run_task_7()
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

def run_task_3():
    hop = HopfieldNetwork(25)

    # 5
    pattern1 = np.array([[1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1]])
    # 7
    pattern2 = np.array([[1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1]])
    # H
    pattern3 = np.array([[1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1]])


    hop.add_pattern_to_memory(pattern1)
    hop.add_pattern_to_memory(pattern2)
    hop.add_pattern_to_memory(pattern3)
    hop.train_patterns()

    # 5
    pattern1_corrupted = np.array([[1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1],
                         [1, 1, 1, 1, 1]])
    # 7
    pattern2_corrupted = np.array([[1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 1, 0, 1],
                         [0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 1]])
    # H
    pattern3_corrupted = np.array([[1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 1, 0, 1],
                         [1, 0, 0, 0, 1]])


    hop.add_corrupted_pattern(pattern1_corrupted)
    hop.add_corrupted_pattern(pattern2_corrupted)
    hop.add_corrupted_pattern(pattern3_corrupted)

    hop.recover_patterns(synchronous=False)

def run_task_4():
    ql = QLearning(grid_size=5, num_episodes=5, num_holes=3, learning_rate=0.1, gamma=0.9, epsilon=0.1)
    history = ql.train()
    ql.replay_training(history, True)

    ql = QLearning(grid_size=5, num_episodes=10, num_holes=3, learning_rate=0.1, gamma=0.9, epsilon=0.1)
    history = ql.train()
    ql.replay_training(history, True)

    ql = QLearning(grid_size=5, num_episodes=100, num_holes=3, learning_rate=0.1, gamma=0.9, epsilon=0.1)
    history = ql.train()
    ql.replay_training(history, True)

def run_task_6():
    # 1
    lsystem = LSystem(axiom="F+F+F+F",rule="F+F-F-FF+F+F-F",angle=90, num_of_nesting=3)
    lsystem.compute()
    lsystem.perform_instructions()
    lsystem.visualize()

    # 2
    lsystem = LSystem(axiom="F++F++F", rule="F+F--F+F", angle=60, num_of_nesting=3)
    lsystem.compute()
    lsystem.perform_instructions()
    lsystem.visualize()

    # 3
    # PI/7 radians to degrees = 25.714
    lsystem = LSystem(axiom="F", rule="F[+F]F[-F]F", angle=25.714, num_of_nesting=3)
    lsystem.compute()
    lsystem.perform_instructions()
    lsystem.visualize()

    # 4
    # PI/8 radians to degrees = 22.5
    lsystem = LSystem(axiom="F", rule="FF+[+F-F-F]-[-F+F+F]", angle=22.5, num_of_nesting=3)
    lsystem.compute()
    lsystem.perform_instructions()
    lsystem.visualize()

def run_task_7():
    ifs = IFS(model_no=1)
    ifs.compute_transformations()
    ifs.visualize()

    ifs = IFS(model_no=2)
    ifs.compute_transformations()
    ifs.visualize()

if __name__ == "__main__":
    choose_task()