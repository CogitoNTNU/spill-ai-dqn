import gym
import random
import numpy as np
import time

TRAINING_EPISODES = 400 # Episodes with decaying EPSILON
STABLE_EPISODES = 50 # Episodes with EPSILON = MIN_EPSILON
RENDER_EPISODES = 5 # Episodes with EPSILON = 0

ALPHA = 0.1
GAMMA = 0.98

EPSILON = 0.9
MIN_EPSILON = 0.05

# Epsilon will decay with a constant rate towards the minimal value
EPSILON_DECAY = (EPSILON-MIN_EPSILON)/(TRAINING_EPISODES-STABLE_EPISODES)

DISCRETIZATION = [5,1,5,1,20,20]


env = gym.make('Acrobot-v1')


linspaces = []

high_space = env.observation_space.high

for i in range(len(DISCRETIZATION)):
    high = high_space[i]
    linspaces.append(np.linspace(-high,high,DISCRETIZATION[i]-1))


def discret_state(state):

    disc = []

    for i in range(len(state)):

        disc.append(np.digitize(state[i], linspaces[i]))

    return disc[0],disc[1],disc[2],disc[3],disc[4],disc[5]


successes = 0


def run_episode(should_render):

    global EPSILON
    global successes

    state = env.reset()

    reward = 0
    terminated = False
    steps = 0

    accu = 0

    if should_render:
        EPSILON = 0

    while not terminated:

        state_indices = discret_state(state)

        q_vals = q_table[state_indices]

        if random.uniform(0,1) < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_vals)

        step_result = env.step(action)

        next_state, reward, terminated, info = step_result

        if reward == 0:
            successes+=1

        steps+=1

        accu += reward

        max_val = np.max(q_table[discret_state(next_state)])

        current_val = q_table[(state_indices + (action,))]

        q_table[(state_indices + (action,))] = current_val + ALPHA*(reward + GAMMA * max_val - current_val)
        state = next_state

        if should_render:
            env.render()

    if EPSILON > MIN_EPSILON:
        EPSILON -= EPSILON_DECAY



q_table = np.zeros([
    DISCRETIZATION[0],
    DISCRETIZATION[1],
    DISCRETIZATION[2],
    DISCRETIZATION[3],
    DISCRETIZATION[4],
    DISCRETIZATION[5],
    env.action_space.n
])

EPISODE_CHUNK = 10

for i in range(1,TRAINING_EPISODES + RENDER_EPISODES+1):


    if (i%EPISODE_CHUNK)==0:
        print(f"Episode {i-9}-{i}:", f"{100*successes/EPISODE_CHUNK}%", f"ε={round(EPSILON,3)}")
        successes = 0

    run_episode(i > TRAINING_EPISODES)


env.close()
