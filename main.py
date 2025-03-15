"""
This script implements a Monte Carlo Tree Search (MCTS) algorithm to solve 
reinforcement learning problem on Gymnasium framework (tested on CartPole-v1).
 
Configurations are imported from the config file.

Implementation Details:
---------------------
- Uses UCB1 formula for node selection with exploration constant from config
- Performs random rollouts to estimate node values
- Detaches parent references to conserve memory
- Selects actions based on visit counts (N) rather than value estimates (T)
- Configurable parameters via external config file

Author: M.I. Khan
Date: 15 March 2025
"""

import numpy as np
import gymnasium as gym
import random
from copy import deepcopy  
from math import *
import config

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------
GAME_NAME = config.GAME_NAME

# Create a temporary environment to extract basic properties
env = gym.make(GAME_NAME)
GAME_ACTIONS = env.action_space.n
GAME_OBS = env.observation_space.shape[0]

print(
    "In the "
    + GAME_NAME
    + " environment there are: "
    + str(GAME_ACTIONS)
    + " possible actions."
)
print(
    "In the "
    + GAME_NAME
    + " environment the observation is composed of: "
    + str(GAME_OBS)
    + " values."
)

env.reset()
env.close()

# Exploring constant from the config file
c = config.c

# -----------------------------------------------------------------------------
# Monte Carlo Tree Search Implementation
# -----------------------------------------------------------------------------
class Node:
    def __init__(self, game, done, parent, observation, action_index):
        self.child = None  # Dict for child nodes, key = action index
        self.T = 0  # Total rewards accumulated from MCTS explorations
        self.N = 0  # Total visit count
        self.game = game  # Game environment state
        self.observation = observation  # Observations from the environment
        self.done = done  # Game termination flag
        self.parent = parent  # Parent node
        self.action_index = action_index  # Action index that lead to this node

    def getUCBscore(self) -> float:
        """
       Calculate UCB1 score for the node, which is used to select the node 
       during selection process.
        """
        # Unexplored nodes have maximum UCB value so exploration is prefered
        if self.N == 0:
            return float("inf")

        # Getting the parent node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        # using the UCB formula on a node
        return (self.T / self.N) + c * sqrt(log(top_node.N) / self.N)

    def detach_parent(self):
        """Detach the node from its parent (save memory)."""
        del self.parent
        self.parent = None

    def create_child(self):
        """
        Creating a child node for each possible action of the game
        """

        if self.done:
            return

        child_dict = {}
        for i in range(GAME_ACTIONS):
            # Create a copy of the current environment to simulate the action
            game_copy = deepcopy(self.game)
            # For Gymnasium, step returns observation, reward, done, truncated, and info
            observation, reward, done, truncated, info = game_copy.step(i)
            # Consider the episode finished if either done or truncated is True
            done = done or truncated
            # Create a new child node based on the simulated action result
            child_node = Node(game_copy, done, self, observation, i)
            child_dict[i] = child_node
        self.child = child_dict

    def rollout(self) -> int:
        """
        It is a random play from a copy of the environment of the current node using random moves.
        This will give us a value for the current node. (more rollouts will lead to more reliable results)
        """

        if self.done:
            return 0

        v = 0
        done = False
        new_game = deepcopy(self.game)
        while not done:
            action = new_game.action_space.sample()
            observation, reward, done, truncated, info = new_game.step(action)
            v = v + reward
            if done:
                new_game.reset()
                new_game.close()
                break
        return v

    def explore(self):
        """
        Performs a tree exploration from the current node in a Monte Carlo Tree Search (MCTS) context.
        This method navigates the tree by recursively selecting the child node with the maximum UCB1 score,
        ensuring that the most promising nodes are further explored. Once a leaf node is reached, the method checks
        if the node has been sufficiently visited:
            - If the node's visit count is low, it performs a rollout from that node to evaluate it.
            - If the node has been visited at least once, it expands the node by creating its children and then
              randomly selects one of the newly created children to perform the rollout.
        After the rollout, the visit count (N) and total accumulated reward (T) of the current node are updated.
        The method then backpropagates the reward up to the root by updating the statistics of all ancestors.
        Returns:
            None
        """
        """
        From the current node, recursively pick the children which maximises the value according to the UCB1 formula
        """

        # find a leaf node by choosing nodes with max U.

        current = self

        while current.child:

            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [a for a, c in child.items() if c.getUCBscore() == max_U]
            if len(actions) == 0:
                print("error zero length", max_U)
            action = random.choice(actions)
            current = child[action]

        # play a random game, or expand if needed

        if current.N < 1:
            current.T = current.T + current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.T = current.T + current.rollout()

        current.N += 1

        # update statistics and backpropogation

        parent = current

        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T

    def next_action(self) -> tuple:
        """
        Selects the next action based on the child node with the highest visit count.
        This function examines the collection of child nodes and determines the one with the maximum visit count (N).
        If multiple child nodes share the maximum visit count, one is selected at random.
        Raises:
            ValueError: If the game has already ended (self.done is True).
            ValueError: If no child node is available when the game has not ended.
        Returns:
            tuple: A tuple containing:
                - The selected child node.
                - The corresponding action index associated with that child node.
        """
        """
        
        """

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError("no child found and game has not ended")

        child = self.child

        max_N = max(node.N for node in child.values())

        max_children = [c for a, c in child.items() if c.N == max_N]

        if len(max_children) == 0:
            print("error zero length ", max_N)

        max_child = random.choice(max_children)

        return max_child, max_child.action_index


# -----------------------------------------------------------------------------
# MCTS Agent and Training
# -----------------------------------------------------------------------------
MCTS_POLICY_EXPLORE = config.MCTS_POLICY_EXPLORE


def Policy_player_MCTS(mytree):
    '''
    Execute the MCTS policy to select the next action to take in the environment.
    
    
    Args:
        mytree (Node): Current root node of the MCTS tree
        
    Returns:
        tuple: (next_node, action)
            - next_node: The selected child node that becomes the new root
            - action: The action index to take in the actual environment
    '''
    for i in range(MCTS_POLICY_EXPLORE):
        mytree.explore()
    next_node, action = mytree.next_action()
    next_node.detach_parent()
    return next_node, action


episodes = config.EPISODES
rewards = []
moving_average = []

# Run training for specific number of episodes
for e in range(episodes):
    reward_e = 0
    game = gym.make(GAME_NAME)
    done = False
    timestep = 0
    observation, _ = game.reset()

    # Initailise the MCTS tree with the initial state
    new_game = deepcopy(game)
    mytree = Node(new_game, False, None, observation, 0)

    print("episode number: " + str(e + 1))

    # Episode loop
    while not done:
        # Use MCTS to select action in the actual environment
        mytree, action = Policy_player_MCTS(mytree)

        # Take the selected action in the actual environment
        observation, reward, done, _, truncated = game.step(action)
        reward_e += reward
        timestep += 1

        if timestep >= 200:
            done = True

        if done:
            print("reward_e :" + str(reward_e))
            game.close()
            break
    
    # Track episode performance metrics
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))
