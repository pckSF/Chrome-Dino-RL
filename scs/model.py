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

    A deep q-learning agent which utilises two neural networks, one for q-value
    predicion used in action prediction and one for the generation of the target
    q-values used for updating the prediction neural network. As the target
    network is only updated to the current state of the prediction network after
    every few updates of the prediction network, its parameters do not directly
    influence its own targets during the updated process. This reduces oscilation
    and stabilises the training procedure.

    Attributes:
            _memory: The replay memory class instance in use.
            _gamma: The discount factor for delayed rewards.
            _epsilon: The probability for a random exploratory action at a timestep.
            _init_epsilon: The initial (max) value of the probability for an
                exploration step in a timestep durong the training process.
            _min_epsilon: The minimum value of epsilon, lowest exploration step
                probability during the training process.
            _qnn_p: The predictive q-value neural network used to predict the
                q-values based on which timesetp action decisions are made.
            _qnn_t: The target q-value neural network used to generate the target
                q-values during training of the prediction q-value neural network.
                is updated based on a frequency parameter which is passed to the
                training method.
            _interface: The interface class instance which is used to interact
                with the chrome-dino game running in the browser.
            score_history: The history of scores for each episode in the training
                process.
            loss_history: The history of losses generated during each update of
                the prediction q-value neural network.
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
    ) -> None:
        """Initialises DinoAgent class with arguments.

        Initialises the agent by compiling the neural networks, and creating
        the target neural network as a copy of the prediction network. Starts the
        interface and sleeps for one second allowing the interface to finalise
        its initialisation.

        Args:
            memory_capacoty: The maximum capacity of the rolling window experience
                replay memory.
            batchsize: Number of timesteps sampled for each update batch.
            gamma: he discount factor for delayed rewards.
            init_epsilon: The initial (max) value of the probability for an
                exploration step in a timestep durong the training process.
            min_epsilon: The minimum value of epsilon, lowest exploration step
                probability during the training process.
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
        """Initialies the qnns; can also be used to reset the nn parameters"""
        self._qnn_p.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            ),
            loss="mse",
        )
        self._qnn_t.set_weights(self._qnn_p.get_weights())

    def reset_model(self) -> None:
        """Resets all model parameters and clears the histories"""
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
        """Restarts the interface which.

        Restarting the interface from time to time is recommended to avoid
        potential crashes of the webdriver during long training processes.
        """
        self._interface.close()
        time.sleep(5)
        self._interface = Interface()
        time.sleep(5)

    def _memory_add_timestep(self, timestep: Timestep) -> None:
        """Adds the data from one timestep to the replay memory"""
        self._memory.add_timestep(timestep)

    def _update(self) -> float:
        """Updates the prediction qnn etc.

        Updates the prediction neural network by sampling a batch from the
        experience replay memory, if it is possible to do so.
        Generates the target q-values by using the q-value predictions from the
        target qnn for the states in t+1. The utilisation of an own target
        qnn, which is updated to match the state of the prediction qnn in
        intervals, stabilises the models convergence.
        """
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
        """
        Updates the target qnn with by setting its parameters to the current
        parameter values of the prediction qnn.
        """
        if self._memory.batch_possible:
            self._qnn_t.set_weights(self._qnn_p.get_weights())
            print(f"{datetime.now()}: Updated target qnn")

    def _step(self) -> bool:
        """Performs one step in the environment

        Performs one action (epsilon-greedy) and adds the timestep
        data to the experience replay memory. In case the step is exploratory,
        an action is sampled at random, else the action with the highest q-value
        is chosen.

        Returns:
            terminal: Boolean value indicating whether the current state
                is a terminal state or not.
        """
        q_values = self._qnn_p.predict(self.state_t[np.newaxis, :])
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
        """
        Plays through one episode in the environment and adds the episodes score
        to the score history.
        """
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
        """Trains the model for episodes epsiodes.

        Trains the model for a number of episodes and prints the progress as well
        as some core metrics for each episode.
        Updates the target qnn based on the passed update_frequency value.
        Also restarts the interface every 500 episode to keep the selenium
        webdriver from crashing (As, based on a few tests, the driver might crash
        after being active for a long time).

        Args:
            episode: The number of episoedes in the training loop.
            contuinue_training: A boolean flag that decides whether the training
                should be continued or if all parameters should be reset before
                starting the training loop.
            update_frequency: The number of episodes between updates of the target
                qnn to to the current state of the prediction qnn.
        """
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

    def print_performance(self) -> None:
        """Plots the score as well as the loss history"""
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
