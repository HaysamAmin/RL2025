"""
assignment3_agent_tf.py

This file defines a DQNAgent class implemented with TensorFlow/Keras
for playing Atari Pong.

Sections:
    1. ReplayBuffer class
       - Stores experience tuples (s, a, r, s', done).
    2. Q-network builder (Keras model)
       - Convolutional neural network mapping state -> Q-values.
    3. DQNAgent class
       - Handles action selection (epsilon-greedy),
         experience storage, sampling,
         and training steps (updating the online network)
         plus periodic target network updates.
"""

import numpy as np
import random
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ======================
# 1. EXPERIENCE REPLAY
# ======================

class ReplayBuffer:
    """
    Simple circular replay buffer for DQN.

    Stores transitions:
        (state, action, reward, next_state, done)

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        All items are stored as-is; we convert to NumPy arrays
        later when sampling batches.

        Parameters
        ----------
        state, next_state : np.ndarray
            Processed states (frames or stacked frames).
        action : int
            Action taken.
        reward : float
            Reward received.
        done : bool
            Whether the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch of transitions.

        Returns
        -------
        states, actions, rewards, next_states, dones : np.ndarray
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.concatenate(states, axis=0),      # (B, H, W, C)
            np.array(actions, dtype=np.int32),   # (B,)
            np.array(rewards, dtype=np.float32), # (B,)
            np.concatenate(next_states, axis=0), # (B, H, W, C)
            np.array(dones, dtype=np.float32)    # (B,)
        )

    def __len__(self):
        """
        Return current number of stored transitions.
        """
        return len(self.buffer)


# ===========================
# 2. Q-NETWORK (Keras Model)
# ===========================

def build_q_network(input_shape, num_actions):
    """
    Build a CNN-based Q-network for Atari Pong using Keras.

    Architecture (simple DQN style):
        - Conv2D(32, 8x8, stride 4) + ReLU
        - Conv2D(64, 4x4, stride 2) + ReLU
        - Conv2D(64, 3x3, stride 1) + ReLU
        - Flatten
        - Dense(512) + ReLU
        - Dense(num_actions)  # Q-values

    Parameters
    ----------
    input_shape : tuple
        Shape of one state, e.g. (H, W, C) = (80, 80, 1 or 4).
    num_actions : int
        Number of discrete actions in the environment.

    Returns
    -------
    keras.Model
        Compiled Q-network.
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=8, strides=4, activation="relu")(inputs)
    x = layers.Conv2D(64, kernel_size=4, strides=2, activation="relu")(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(num_actions, activation=None)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Mean squared error loss for Q-learning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse"
    )
    return model


# =================
# 3. DQN AGENT CLASS
# =================

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for Atari Pong.

    Responsibilities:
        - Maintain online and target Q-networks.
        - Store transitions in a replay buffer.
        - Select actions (epsilon-greedy policy).
        - Perform training steps using sampled batches.
        - Periodically update the target network.

    Parameters
    ----------
    state_shape : tuple
        Shape of one processed state (H, W, C).
    num_actions : int
        Number of actions available in the environment.
    """

    def __init__(
        self,
        state_shape,
        num_actions,
        gamma=0.99,
        buffer_capacity=100_000,
        batch_size=32,
        min_replay_size=10_000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_frames=1_000_000,
        target_update_freq=10_000
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions

        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size

        # Epsilon-greedy scheduling parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.frame_idx = 0

        # Target network update frequency (in environment steps).
        self.target_update_freq = target_update_freq

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Build online and target Q-networks
        self.online_net = build_q_network(self.state_shape, self.num_actions)
        self.target_net = build_q_network(self.state_shape, self.num_actions)

        # Initialize target network weights to match online network
        self.target_net.set_weights(self.online_net.get_weights())

    # -----------------------------
    # Epsilon-greedy action selection
    # -----------------------------

    def get_epsilon(self):
        """
        Linearly decay epsilon from epsilon_start to epsilon_end
        over epsilon_decay_frames environment steps.

        Returns
        -------
        float
            Current epsilon value.
        """
        fraction = min(1.0, self.frame_idx / self.epsilon_decay_frames)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            Current state, expected shape (1, H, W, C).

        Returns
        -------
        int
            Selected action index.
        """
        epsilon = self.get_epsilon()
        self.frame_idx += 1  # increment step count

        if random.random() < epsilon:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)

        # Exploit: choose action with highest predicted Q-value
        q_values = self.online_net.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    def select_greedy_action(self, state):
        """
        Select the best action (no exploration).

        Used for evaluation / watching the trained policy.
        """
        q_values = self.online_net.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    # -----------------------------
    # Replay buffer interaction
    # -----------------------------

    def store_transition(self, state, action, reward, next_state, done):
        """
        Save a transition into the replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    # -----------------------------
    # Training / optimization
    # -----------------------------

    def can_train(self):
        """
        Check if replay buffer has enough samples to start training.
        """
        return len(self.replay_buffer) >= self.min_replay_size

    def train_step(self):
        """
        Perform one training step on a sampled batch from the replay buffer.

        Steps:
            1. Sample a batch of transitions.
            2. Compute target Q-values.
            3. Compute current Q-values for the actions taken.
            4. Minimize MSE loss between targets and current Q-values.

        This is the core update for the online Q-network.
        """
        if not self.can_train():
            # Not enough data yet, skip training
            return None

        # Sample transitions from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Predict Q-values for next_states using target network
        next_q_values = self.target_net.predict(next_states, verbose=0)  # (B, num_actions)
        max_next_q = np.max(next_q_values, axis=1)                       # (B,)

        # Compute target Q-values using Bellman equation:
        # target = r + gamma * max_a' Q_target(s', a') * (1 - done)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        # Predict current Q-values from online network
        q_values = self.online_net.predict(states, verbose=0)

        # Create a copy so we only change Q(s,a) for the taken actions
        q_values_target = np.array(q_values, copy=True)

        # Update the Q-value targets for the actions taken in each state
        for i, action in enumerate(actions):
            q_values_target[i, action] = targets[i]

        # Train the online network on this batch
        loss = self.online_net.train_on_batch(states, q_values_target)

        # Periodically update the target network to match online network
        if self.frame_idx % self.target_update_freq == 0:
            self.target_net.set_weights(self.online_net.get_weights())

        return loss

    # -----------------------------
    # Utility methods
    # -----------------------------

    def save(self, filepath: str):
        """
        Save the online Q-network to a .keras file.

        Note: We avoid .h5 as requested, and use the newer Keras format.
        """
        self.online_net.save(filepath)

    def load(self, filepath: str):
        """
        Load weights from a saved .keras model into the online network,
        and copy them into the target network as well.
        """
        loaded_model = keras.models.load_model(filepath)
        self.online_net.set_weights(loaded_model.get_weights())
        self.target_net.set_weights(loaded_model.get_weights())

    def eval_mode(self):
        """
        Set the model in evaluation mode.

        For Keras this is mostly a no-op, but we keep this method
        for a consistent agent interface.
        """
        # In PyTorch we would do model.eval(), but in Keras,
        # predict() already runs in inference mode.
        pass
