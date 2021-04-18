import random
from dataclasses import dataclass
from typing import List, Tuple

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

    state: np.ndarray
    action: int
    reward: int
    q_values: np.ndarray
    next_state: np.ndarray
    terminal: bool


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

    def __init__(self, capacity: int, batchsize: int):
        """Initialises the empty experience replay memory with args"""
        self._capacity: int = capacity
        self._capacity_reached: bool = False
        self._batchsize: int = batchsize
        self.batch_possible: bool = False
        self._buffer: List[Timestep] = []

    def add_timestep(self, timestep_buffer: Timestep) -> None:
        """
        Adds a single timestep to the memory and sets the _batch_possible as well
        as the _capacitzy_reached flags if conditions are met.
        Keeps rolling window of the last _capacity number of timeslots once
        _caüacity is reached.
        """
        self._buffer.append(timestep_buffer)
        if not self.batch_possible:
            if len(self._buffer) >= self._batchsize:
                self.batch_possible = True
        if not self._capacity_reached:
            if len(self._buffer) >= self._capacity:
                self._capacity_reached = True
        else:
            del self._buffer[0]

    def sample_batch(self) -> Tuple[int, List[Timestep]]:
        """Returns a random uniform sampled batch of batchsize"""
        return self._batchsize, random.sample(self._buffer, self._batchsize)

    def reset(self) -> None:
        """Resets the replay memory"""
        self._buffer.clear()
        self._capacity_reached = False
        self.batch_possible = False
