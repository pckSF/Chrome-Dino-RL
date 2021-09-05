from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Timestep:
    """Dataclass that holds the data of one q-learning timestep.

    Attributes:
        state: The starting state of the timestep
        action: The action performed in the timestep
        reward: The reward obtained in the timestep
        q_values: The predicted q_values of the timestep
        next_state: The state resulting from 'action' in 'state'
        terminal: Timestep ends in terminal state
    """

    state: None | np.ndarray = None
    action: None | int = None
    reward: None | int = None
    q_values: None | np.ndarray = None
    next_state: None | np.ndarray = None
    terminal: None | bool = None


class ReplayMemory:
    """Simple experience replay memory.

    Experience replay memory for q-learning agents that holds data from timestep
    experience and generates batches of predefined size by randomly sampling
    from the buffer.

    Attributes:
        _capacity: Maximum number of timesteps to be stored in the memory
        _capacity_reached: Flag that indicates if the memory is filled and the
            and becomes a rolling window of the last _capacity timesteps
        _batchsize: Number of timesteps returned in a sampe batch
        _buffer: The list that stores the timestep data.
    """

    def __init__(self, capacity: int, batchsize: int) -> None:
        """Initialises the empty experience replay memory with args"""
        self._capacity: int = capacity
        self._capacity_reached: bool = False
        self._batchsize: int = batchsize
        self.batch_possible: bool = False
        self._buffer: list[Timestep] = [Timestep() for _ in range(capacity)]
        self._data_index: int = 0

    def add_timestep(self, timestep: Timestep) -> None:
        """
        Adds a single timestep to the memory and sets the batch_possible as well
        as the _capacitzy_reached flags if conditions are met.
        Keeps rolling window of the last _capacity number of timeslots once
        _capacity is reached.
        """
        self._buffer[self._data_index] = timestep
        self._data_index += 1
        if not self.batch_possible:
            if self._data_index >= self._batchsize:
                self.batch_possible = True
        if self._data_index >= self._capacity:
            self._capacity_reached = True
            self._capacity = 0

    def sample_batch(self) -> tuple[int, list[Timestep]]:
        """Returns a random uniform sampled batch of batchsize"""
        if not self._capacity_reached:
            batch = random.sample(self._buffer[: self._data_index], self._batchsize)
        else:
            batch = random.sample(self._buffer, self._batchsize)
        return self._batchsize, batch

    def reset(self) -> None:
        """Resets the replay memory"""
        self._data_index = 0
        self._capacity_reached = False
        self.batch_possible = False
