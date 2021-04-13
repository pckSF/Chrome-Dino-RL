import random

import numpy as np
import pytest

from scs.memory import ReplayMemory, Timestep

random.seed(12345)


@pytest.fixture
def replay_memory() -> ReplayMemory:
    return ReplayMemory(capacity=10, batchsize=5)


@pytest.mark.parametrize(
    "timesteps, ex_buffer, ex_capacity_reached, ex_batch_possible",
    [
        pytest.param(
            (
                Timestep(
                    state=np.array([i]),
                    action=i,
                    reward=i,
                    q_values=np.array([i]),
                    next_state=np.array([i]),
                    terminal=True,
                )
                for i in range(2)
            ),
            (
                Timestep(
                    state=np.array([i]),
                    action=i,
                    reward=i,
                    q_values=np.array([i]),
                    next_state=np.array([i]),
                    terminal=True,
                )
                for i in range(2)
            ),
            False,
            False,
            id="Number of timesteps < batchsize",
        ),
        pytest.param(
            (
                Timestep(
                    state=np.array([i]),
                    action=i,
                    reward=i,
                    q_values=np.array([i]),
                    next_state=np.array([i]),
                    terminal=True,
                )
                for i in range(5)
            ),
            (
                Timestep(
                    state=np.array([i]),
                    action=i,
                    reward=i,
                    q_values=np.array([i]),
                    next_state=np.array([i]),
                    terminal=True,
                )
                for i in range(5)
            ),
            False,
            True,
            id="Number of timesteps == batchsize",
        ),
        pytest.param(
            (
                Timestep(
                    state=np.array([i]),
                    action=i,
                    reward=i,
                    q_values=np.array([i]),
                    next_state=np.array([i]),
                    terminal=True,
                )
                for i in range(10)
            ),
            (
                Timestep(
                    state=np.array([i]),
                    action=i,
                    reward=i,
                    q_values=np.array([i]),
                    next_state=np.array([i]),
                    terminal=True,
                )
                for i in range(10)
            ),
            True,
            True,
            id="Number of timesteps == capacity",
        ),
        pytest.param(
            (
                Timestep(
                    state=np.array([i]),
                    action=i,
                    reward=i,
                    q_values=np.array([i]),
                    next_state=np.array([i]),
                    terminal=True,
                )
                for i in range(14)
            ),
            (
                Timestep(
                    state=np.array([i]),
                    action=i,
                    reward=i,
                    q_values=np.array([i]),
                    next_state=np.array([i]),
                    terminal=True,
                )
                for i in range(4, 14)
            ),
            True,
            True,
            id="Number of timesteps > capacity",
        ),
    ],
)
def test_add_timestep(
    replay_memory, timesteps, ex_buffer, ex_capacity_reached, ex_batch_possible
):
    for ts in timesteps:
        replay_memory.add_timestep(ts)
    assert replay_memory._buffer == list(ex_buffer)
    assert replay_memory._capacity_reached is ex_capacity_reached
    assert replay_memory.batch_possible is ex_batch_possible


def test_sample_batch(replay_memory):
    test_data = (
        Timestep(
            state=np.array([i]),
            action=i,
            reward=i,
            q_values=np.array([i]),
            next_state=np.array([i]),
            terminal=True,
        )
        for i in range(10)
    )
    replay_memory._buffer = list(test_data)
    result = replay_memory.sample_batch()
    expected = (
        5,
        [
            Timestep(
                state=np.array([i]),
                action=i,
                reward=i,
                q_values=np.array([i]),
                next_state=np.array([i]),
                terminal=True,
            )
            for i in [6, 0, 4, 9, 2]
        ],
    )
    assert result == expected


def test_reset(replay_memory):
    replay_memory._buffer = list(range(10))
    replay_memory._capacity_reached = True
    replay_memory.batch_possible = True
    replay_memory.reset()
    assert len(replay_memory._buffer) == 0
    assert replay_memory._capacity_reached is False
    assert replay_memory.batch_possible is False
