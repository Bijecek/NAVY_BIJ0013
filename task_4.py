# BIJ0013 - task_4
# Cílem bylo vytvořit Q-Learning model, kdy agent hledá sýr a minimalizovat uraženou vzdálenost bez toho, aby vstoupil do díry
# - Workflow:
#     - Inicializace
#         - vytváří se matice, která je inicializovaná na 0 - tato matice má na každé pozici [X,Y] 4 hodnoty - jedna hodnota pro každý pohyb
#         - vytváří se díry na náhodných pozicích kromě počáteční a konečné
#
#     - Trénink modelu
#         - trénování probíhá v N epizodách, na začátku každé epizody se nastaví počáteční "defaultní" hodnoty např. pro aktuální pozici
#         - epizoda probíhá do té doby, dokud nenarazíme na cíl - sýr nebo nevstoupíme do díry
#             - v každé epizodě se určí možné kroky, které mohou na dané pozici nastat
#             - je zde šance, že se vybere náhodný krok (epsilon) nebo krok, který má největší Q-hodnotu v matici (na aktuální pozici)
#             - po vybrání kroku se vypočítá na jaké pozici se momentálně agent nachází
#             - následuje výpočet odměny (obsahuje i malou penalizaci za pohyb)
#             - následně dochází k aktualizaci matice a změny aktuální pozice
#     - Zobrazení výsledků
#         - výsledky tréninku se vykreslí v animaci (defaulně se zobrazuje jen poslední úspěšná epizoda)
#
#  Tento Q-learning jsem trénoval ve třech "nastaveních" - 5, 10 a 100 epizod na 5x5 matici (mřížce) se třemi dírami
#
#    -> 5 epizod nebylo dostatečných, agent v každé z nich vstoupil do díry.
#
#    -> Při 10 epizodách již vidíme, že se agent ve 13 krocích dostal k cíli
#
#    -> Při 100 již agent dosáhl nejkratší možné cesty k sýru (výsledku).
#
# NOTE: Modrá barva představuje aktuální pozici agenta v mřížce, červenou jsou označeny díry a zelená je cílová pozice

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class QLearning:
    def __init__(self, grid_size, num_episodes, num_holes, learning_rate, gamma, epsilon):
        # Velikost čtvercové mřížky
        self.grid_size = grid_size

        np.random.seed(123)
        # Generování num_holes děr
        self.holes = set()
        while len(self.holes) < num_holes:
            hole = (np.random.randint(grid_size), np.random.randint(grid_size))
            if hole != (0, 0) and hole != (grid_size - 1, grid_size - 1):
                self.holes.add(hole)

        self.holes = list(self.holes)

        self.board = None
        self.current_position = None

        # Nastavení výherní pozice na pravý dolní roh
        self.winning_position = (grid_size-1, grid_size-1)

        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.fill_board()

        self.all_actions = ["right", "left", "up", "down"]

        self.position_history = []
        self.episode_history = []

    # Metoda pro naplnění matice nulami
    def fill_board(self):
        self.board = np.zeros((self.grid_size, self.grid_size, 4))

        # Nastavení defaultní pozice na levý horní roh
        self.current_position = (0,0)

    # Metoda pro určení všech možných akcí pro danou pozici - tzn. jak se můžeme pohnout
    def calculate_action(self, state):
        action = []
        pos_x, pos_y = state

        if pos_y < self.grid_size-1:
            action.append("right")
        if pos_y > 0:
            action.append("left")
        if pos_x > 0:
            action.append("up")
        if pos_x < self.grid_size - 1:
            action.append("down")

        return action

    # Pomocná metoda pro určení změny pozice pro danou akci
    def calculate_next_state(self, state, action):
        pos_x, pos_y = state

        match action:
            case "right":
                pos_y += 1
            case "left":
                pos_y -= 1
            case "up":
                pos_x -= 1
            case "down":
                pos_x += 1

        return pos_x, pos_y

    # Aktualizujeme matici pomocí dané akce a odměny
    def update_q_values(self, action_index, reward, next_state):
        return ((1 - self.learning_rate) * self.board[self.current_position[0], self.current_position[1], action_index]
                + self.learning_rate * (reward + self.gamma * np.max(self.board[next_state[0], next_state[1]])))

    # Hlavní trénovací smyčka
    def train(self):
        for episode in range(1, self.num_episodes+1):

            # Resetujeme aktuální pozici a odměnu na začátku každé epizody
            self.current_position = (0,0)
            total_reward = 0

            # Vytváříme si pomocné proměnné pro následné vykreslování
            position_history = [self.current_position]
            reward_history = [total_reward]
            done = False

            # S trénováním končíme, pokud jsme na "pozici sýru" nebo jsme spadli do díry
            while self.current_position != self.winning_position and not done:
                # Určíme všechny možné akce pro aktuální pozici
                actions = self.calculate_action(self.current_position)

                # Přidáváme náhodnou akci - epsilon greedy přístup
                if np.random.random() < self.epsilon:
                    action = np.random.choice(actions)
                else:
                    # Získáme indexy možných akcí (0 - 3) a následně získáme hodnoty z matice na těchto indexech
                    possible_action_indexes = [self.all_actions.index(action) for action in actions]
                    action_values = [self.board[self.current_position[0], self.current_position[1], index] for index in possible_action_indexes]

                    # Vybereme největší hodnotu z pole action_values - respektive její index v tomto poli,
                    #   následně se jako výsledná akce vybere akce na tomto indexu
                    #       -> například pohyb vpravo
                    action = actions[np.argmax(action_values)]

                next_state = self.calculate_next_state(self.current_position, action)

                # Reward funkce - penalizujeme zbytečný pohyb
                if next_state == self.winning_position:
                    reward = 1
                elif next_state in self.holes:
                    reward = -1
                    done = True
                else:
                    reward = -0.01

                total_reward += reward

                # Aktualizujeme matici pro danou provedenou akci
                action_index = self.all_actions.index(action)
                self.board[self.current_position[0], self.current_position[1], action_index] = self.update_q_values(action_index, reward, next_state)

                self.current_position = next_state
                position_history.append(self.current_position)
                reward_history.append(total_reward)

            # Do vykreslování přidávám pouze ty epizody, které úspěšně našly cíl
            if not done:
                self.episode_history.append((episode, position_history, reward_history))

        if len(self.episode_history) == 0:
            print(f"No successful episode in {self.num_episodes} episodes")
        else:
            _, steps, reward_last_ep = self.episode_history[-1]
            print(f"Reward from episode {self.num_episodes}: {round(reward_last_ep[-1], 2)} in {len(steps)} steps")
            return self.episode_history

    # Metoda pro vizualizaci
    def replay_training(self, training_history, show_only_last):
        if training_history is None:
            return
        if show_only_last:
            training_history = training_history[-1]
        fig, ax = plt.subplots()

        frames = []
        if show_only_last:
            episode, steps, rewards = training_history
            for step_idx, pos in enumerate(steps):
                frames.append(
                    (episode, step_idx, steps[:step_idx + 1], len(steps), rewards[step_idx]))

        else:
            for episode, steps, rewards in training_history:
                for step_idx, pos in enumerate(steps):
                    frames.append(
                        (episode, step_idx, steps[:step_idx + 1], len(steps), rewards[step_idx]))

        def update(frame):
            ax.clear()
            episode, step_idx, path, total_episode_len, cumulative_reward = frame
            grid = np.zeros((self.grid_size, self.grid_size, 3))


            grid[self.grid_size - 1, self.grid_size - 1] = [0, 1, 0]
            for hole in self.holes:
                grid[hole[0], hole[1]] = [1, 0, 0]

            for i, pos in enumerate(path):
                if i == step_idx:
                    grid[pos[0], pos[1]] = [0, 0, 1]


            ax.imshow(grid, interpolation='nearest')
            ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.grid(which="minor", color="grey", linestyle='-', linewidth=2)
            ax.set_title(f"Episode: {episode}, Step: {step_idx + 1}/{total_episode_len}, reward: {round(cumulative_reward, 2)}")

        ani = FuncAnimation(fig, update, frames=frames, interval=500, repeat=False)
        plt.show()




