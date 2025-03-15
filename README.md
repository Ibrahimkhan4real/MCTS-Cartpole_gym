# Monte Carlo Tree Search (MCTS) for Gymnasium Reinforcement Learning

This project implements a Monte Carlo Tree Search (MCTS) algorithm designed to solve reinforcement learning problems within the Gymnasium framework. It has been tested on the CartPole-v1 environment and uses a separate configuration file to set parameters. The code is modular, with a focus on clarity, memory management, and configurable exploration strategies.

---

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Environment Setup](#environment-setup)
- [MCTS Implementation Details](#mcts-implementation-details)
  - [The `Node` Class](#the-node-class)
    - [Initialization](#initialization)
    - [Calculating the UCB Score](#calculating-the-ucb-score)
    - [Memory Management: Detaching Parent](#memory-management-detaching-parent)
    - [Child Node Expansion](#child-node-expansion)
    - [Rollout Function](#rollout-function)
    - [Tree Exploration](#tree-exploration)
    - [Action Selection](#action-selection)
- [Policy and Training Loop](#policy-and-training-loop)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This script uses a Monte Carlo Tree Search algorithm to explore and solve a reinforcement learning task. The MCTS approach leverages random rollouts, the UCB1 formula for node selection, and backpropagation of rewards to guide exploration. The implementation is tailored for Gymnasium environment Cartpole.

---

## Dependencies

- **Gymnasium:** Provides the environment interface (tested with CartPole-v1).
- **NumPy:** Used for numerical operations and statistics (e.g., moving averages).
- **Random:** For random selection during exploration and rollouts.
- **Deepcopy (from `copy`):** To create independent copies of the game state.
- **Math:** Provides mathematical functions such as `sqrt` and `log`.
- **Config:** A configuration file that defines parameters like `GAME_NAME`, exploration constant `c`, number of episodes, and MCTS exploration iterations.

---

## Environment Setup

The script begins by setting up the environment:

1. **Importing the Environment:**
   - The environment is instantiated using `gym.make(GAME_NAME)`, where `GAME_NAME` is imported from the config file.
2. **Extracting Environment Properties:**
   - It retrieves the number of actions available (`GAME_ACTIONS`) and the shape of the observation space (`GAME_OBS`).
3. **Verification:**
   - A temporary environment is created, its properties are printed, and then it is closed to conserve resources.

This preliminary setup ensures that the script is aware of the environment's configuration before running the MCTS algorithm.

---

## MCTS Implementation Details

The core logic is implemented within the `Node` class. Each instance of `Node` represents a state in the game tree, holding the environment state, observation, and the action that led to that state.

### The `Node` Class

#### Initialization

- **Constructor (`__init__`):**
  - Initializes a node with:
    - The current game state (`game`).
    - The observation from the environment (`observation`).
    - A flag indicating if the game has terminated (`done`).
    - A pointer to the parent node (`parent`).
    - The action index that led to this node (`action_index`).
  - Initializes counters:
    - `T`: Total accumulated reward from simulations.
    - `N`: Visit count (number of times the node was explored).
  - Sets up an empty placeholder for child nodes.

#### Calculating the UCB Score

- **`getUCBscore` Method:**
  - **Purpose:** Computes the Upper Confidence Bound (UCB1) score for the node to balance exploration and exploitation.
  - **Details:**
    - Returns infinity for nodes that have not been visited (`N == 0`), ensuring they are explored.
    - Otherwise, uses the formula:  
      \( \text{UCB} = \frac{T}{N} + c \times \sqrt{\frac{\log(\text{parent.N})}{N}} \)
    - The exploration constant `c` is obtained from the config file.

#### Memory Management: Detaching Parent

- **`detach_parent` Method:**
  - **Purpose:** Removes the reference to the parent node to save memory once a node is chosen for the next action.
  - **Operation:** Deletes the parent pointer, ensuring the tree does not retain unnecessary links.

#### Child Node Expansion

- **`create_child` Method:**
  - **Purpose:** Expands the current node by creating a child node for every possible action.
  - **Process:**
    - Iterates over all possible actions (from 0 to `GAME_ACTIONS - 1`).
    - For each action:
      - Creates a deep copy of the current game state.
      - Simulates the action using `game_copy.step(i)`.
      - Considers the episode finished if either `done` or `truncated` is true.
      - Instantiates a new `Node` with the resulting state and adds it to a child dictionary.
    - Attaches the dictionary to `self.child`.

#### Rollout Function

- **`rollout` Method:**
  - **Purpose:** Performs a random simulation (rollout) starting from the current node.
  - **Operation:**
    - If the node is terminal (`done`), returns 0 immediately.
    - Otherwise, repeatedly samples random actions until a terminal state is reached.
    - Accumulates and returns the total reward obtained during the rollout.
    - Closes the game environment after termination to free resources.

#### Tree Exploration

- **`explore` Method:**
  - **Purpose:** Conducts the tree search by recursively selecting child nodes using the UCB1 score.
  - **Steps:**
    1. **Selection:**
       - Navigates down the tree by choosing the child with the highest UCB score until a leaf node (node with no children) is found.
    2. **Expansion/Rollout:**
       - If the node’s visit count (`N`) is low, performs a rollout directly.
       - Otherwise, expands the node using `create_child` and randomly selects one of the newly created child nodes to perform a rollout.
    3. **Backpropagation:**
       - Updates the current node's statistics (`T` and `N`) based on the rollout result.
       - Propagates the accumulated reward and visit count back up the tree to the root.

#### Action Selection

- **`next_action` Method:**
  - **Purpose:** Determines the next action to take in the actual environment based on the MCTS search.
  - **Mechanism:**
    - Evaluates the visit count (`N`) of each child node.
    - Selects the child node with the maximum visit count.
    - If multiple children share the same visit count, randomly selects one.
    - Returns a tuple containing the selected child node and its corresponding action index.
  - **Error Handling:**
    - Raises errors if the game is already terminated or if no children exist.

---

## Policy and Training Loop

### MCTS Policy

- **`Policy_player_MCTS` Function:**
  - **Description:** Acts as the agent’s decision-maker.
  - **Process:**
    - Executes a fixed number of MCTS explorations (set by `MCTS_POLICY_EXPLORE` in the config file).
    - After sufficient exploration, it selects the next best action using the `next_action` method.
    - Detaches the parent reference from the selected node to optimize memory usage.
    - Returns the next node and the action to be executed.

### Training Loop

- **Episode Execution:**
  - For each episode (controlled by `EPISODES` in the config):
    1. A new environment is instantiated and reset.
    2. An initial MCTS tree (`mytree`) is created using the current state.
    3. The agent repeatedly uses the MCTS policy to choose an action.
    4. The chosen action is applied to the actual environment.
    5. The episode continues until termination (or after 200 timesteps as a safeguard).
  - **Metrics:**
    - The reward for each episode is recorded.
    - A moving average of rewards (over the last 100 episodes) is calculated to monitor performance.

---


## Usage
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ibrahimkhan4real/MCTS-Cartpole_gym.git
2. **Install Dependencies:**
   ```bash
   pip install gymnasium numpy
3. **Run the python script:**
   ```bash
   python main.py
   
