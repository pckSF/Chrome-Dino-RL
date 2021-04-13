import random
import time
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scs.dqn import make_dqn
from scs.interface import Interface
from scs.memory import ReplayMemory, Timestep


class DinoAgent:
    """DQN Agent to play the chrome-dino game

    Longer class information....
    Longer class information....

    Attributes:
            memory:
            gamma:
            epsilon:
            min_epsilon:
            qnn_p:
            qnn_t
            interface:
    """

    state_t: np.ndarray
    score_t: int

    def __init__(
        self,
        memory_capacity: int = 500,
        batchsize: int = 16,
        gamma: float = 0.85,
        init_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
    ):
        """Initialises DinoAgent class with arguments

        Args:
                memory_capacoty:
                batchsize:
                gamma:
                init_epsilon:
                min_epsilon:
        """
        self._memory: ReplayMemory = ReplayMemory(memory_capacity, batchsize)
        self._gamma: float = gamma
        self._epsilon: float = init_epsilon
        self._init_epsilon: float = init_epsilon
        self._min_epsilon: float = min_epsilon
        self._qnn_p: tf.keras.Sequential = make_dqn(2, (4, 50, 100))
        self._qnn_t: tf.keras.Sequential = tf.keras.models.clone_model(self._qnn_p)
        self._initialize_qnns()
        self._interface: Interface = Interface()
        self.score_history: List[int] = []
        self.loss_history: List[float] = []
        time.sleep(1)

    def _initialize_qnns(self) -> None:
        """Initialies the qnns also used for reset"""
        self._qnn_p.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            ),
            loss="mse",
        )
        self._qnn_t.set_weights(self._qnn_p.get_weights())

    def reset_model(self) -> None:
        """Resets all model parameters"""
        self._initialize_qnns()
        self._epsilon = self._init_epsilon
        self.score_history.clear()
        self.loss_history.clear()

    def _update_epsilon(self) -> None:
        """Updates the epsilon value"""
        if self._memory.batch_possible:
            self._epsilon = np.max(
                [self._min_epsilon, self._epsilon - self._epsilon / 100]
            )

    def _restart_interface(self) -> None:
        """Restarts the interface to avoid xyz"""
        self._interface.close()
        time.sleep(5)
        self._interface = Interface()
        time.sleep(5)

    def _memory_add_timestep(self, timestep: Timestep) -> None:
        """Adds the data from one timestep to the replay memory"""
        self._memory.add_timestep(timestep)

    def _update(self) -> float:
        """Updates the prediction qnn etc."""
        if not self._memory.batch_possible:
            return 0.0
        batchsize, batch = self._memory.sample_batch()
        states = np.zeros((batchsize, 4, 50, 100))
        targets = np.zeros((batchsize, 2))
        for i, ts in enumerate(batch):
            states[i] = ts.state
            targets[i] = ts.q_values
            if ts.terminal:
                targets[i, ts.action] = -1
            else:
                action_t1 = np.argmax(self._qnn_p.predict(ts.next_state))
                q_values_t1 = self._qnn_t.predict(ts.next_state)[0]
                targets[i, ts.action] = 1 + self._gamma * q_values_t1[action_t1]
        return self._qnn_p.train_on_batch(states, targets)

    def _update_target_qnn(self) -> None:
        """Updates the target qnn with the current values of the prediction qnn"""
        if self._memory.batch_possible:
            self._qnn_t.set_weights(self._qnn_p.get_weights())
            print(f"{datetime.now()}: Updated target qnn")

    def _step(self) -> bool:
        """Performs one step in the environment

        Perform one action (exploring or epsilon-greedy) and return subsequent state
        Stepwise reward is obtained by subtracting the previous step score_p from
        the current score_t
        """
        q_values = self._qnn_p.predict(self.state_t[np.newaxis, :])
        print(f"###################### Q-VALUES: {q_values[0]}")
        if random.random() < self._epsilon:
            action_t = np.random.choice([0, 1])
        else:
            action_t = np.argmax(q_values)
        state, score_t1, terminal = self._interface.action(action_t)
        state_t1 = np.append(state, self.state_t[:-1], axis=0)
        terminal = self._interface.check_crashed()
        self._memory_add_timestep(
            Timestep(
                state=self.state_t,
                action=action_t,
                reward=1,
                q_values=q_values,
                next_state=state_t1[np.newaxis, :],
                terminal=terminal,
            )
        )
        self.state_t, self.score_t = state_t1, score_t1
        return terminal

    def play_episode(self) -> None:
        """Play one episode in the environment"""
        self.state_t = np.zeros((4, 50, 100))
        self.state_t[0] = self._interface.re_start()
        self.score_t = 0
        terminal = False
        while not terminal:
            terminal = self._step()
        self.score_history.append(self._interface.get_score())

    def train(
        self,
        episodes: int,
        continue_training: bool = True,
        update_frequency: int = 5,
    ) -> None:
        """Trains the model for epsiodes"""
        if not continue_training:
            self.reset_model()
        for e in range(episodes):
            self.play_episode()
            loss = self._update()
            self._update_epsilon()
            self.loss_history.append(loss)
            print(
                f"{datetime.now()}: "
                f"Episode {e + 1} of {episodes} completed with with score: "
                f"{self.score_history[-1]}; Updated epsilon to: {self._epsilon}"
            )
            if (e + 1) % update_frequency == 0:
                self._update_target_qnn()
                mean = np.mean(self.score_history[-update_frequency:])
                print(
                    f"{datetime.now()}: "
                    f"Rolling {update_frequency}-episodes mean score: {mean}"
                )
            if (e + 1) % 500 == 0:
                print(f"{datetime.now().time()}: Restarting interface")
                self._restart_interface()

    def print_results(self) -> None:
        """Prints the results"""
        if len(self.score_history) == 0 or len(self.loss_history) == 0:
            raise ValueError(
                "No results to print; score_history and/or loss_history are empty"
            )
        fig, (ax_score, ax_loss) = plt.subplots(2, sharex=True)
        fig.suptitle("Model performance per training episode")
        ax_score.plot(self.score_history, color="blue")
        ax_score.set_ylabel("Score")
        ax_loss.plot(self.loss_history, color="red")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlabel("Episode")
        plt.show()
