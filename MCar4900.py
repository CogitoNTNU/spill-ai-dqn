import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

# Initialize q-table values to 0
discretization = 15
pos_discretization = 15
Q = np.zeros((pos_discretization,discretization, env.action_space.n))    # (states_space, action_space)


# Initialize hyperparameters
gamma = 0.98         # Discount factor used to balance immediate and future reward.
lr = 0.1            # learning rate
epsilon = 0.9         # rate of exploration where 1 is completely random and 0 is deterministic
epsilon_decay = 0.02
# Initialize list for plotting running mean
rewards = []
N = 100     # Number of episodes for the running mean

def running_mean(x, N):
    """function for creating list with running mean"""
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



def get_state(cart_position, tip_speed):
    """Returns index of the given speed state in the Q-table"""
    positions = np.linspace(-1.2,0.6,num=pos_discretization)
    states = np.linspace(-0.07,0.07,num=discretization)   # Discretized table of speeds. 
    for rounded_cart_position in positions:
        if rounded_cart_position >= cart_position:
            cart_index = np.where(positions == rounded_cart_position)[0][0]
            for rounded_tip_speed in states:
                if rounded_tip_speed >= tip_speed:
                    index = np.where(states == rounded_tip_speed)
                    index = index[0][0]
                    return cart_index, index
    print("NONE",cart_position,cart_index)
    print(str(index))
    return None

def better_action(cart_position, tip_speed):
    """Returns index of best action according to Q-table for a given tip speed"""
    possible_actions_in_state = Q[get_state(cart_position, tip_speed)]
    action_of_choice = np.argmax(possible_actions_in_state)
    return action_of_choice

def update_q(state, action, new_state, cart_pos, new_cart_pos, reward=1):
    Q[cart_pos, state, action] = Q[cart_pos, state, action] + lr * (reward + gamma * np.max(Q[new_cart_pos, new_state, :]) - Q[cart_pos, state, action])




for i_episode in range(5000):
    observation = env.reset()
    episode_reward = 0
    

    
    if 4900 < i_episode :
        # Drop exploration
        epsilon = 0
        
    if i_episode%100 == 0 and i_episode != 0:
        #Give some feedback for improvement of the algorithm
        print(i_episode)
        print("best run was", max(rewards[-99:]))
    
        
    for timestep in range(500):
        if i_episode > 4900:
            env.render()
            pass

        #if len(rewards) > 1:

            #if episode_reward >= max(rewards) and episode_reward > -150:
                #Implementing dynamic epsilon so that exploration decays over time.
             #   epsilon -= 0.006
    

        tip_speed = observation[1]
        cart_position = observation[0]

        if random.uniform(0, 1) < epsilon:      # Chooses random action often if epsilon is high
            action = env.action_space.sample()

        else: #Chooses action based on previous exploration
            action = better_action(cart_position, tip_speed)


        
        # Gather information about changes in environment 
        cart_state, state = get_state(cart_position,tip_speed)
        observation, reward, done, info = env.step(action)
        new_tip_speed = observation[1]
        new_cart_pos = observation[0]
        new_cart_state, new_state = get_state(new_cart_pos, new_tip_speed)

        # Update Q-table
        update_q(state, action, new_state, cart_state, new_cart_state, reward)

        episode_reward += reward

        if done:
            #print("Episode finished after {} timesteps".format(timestep+1))
            break
    if i_episode > 0 & i_episode < 4900:
        epsilon = epsilon - epsilon/i_episode
    rewards.append(episode_reward)
print(Q)
print(epsilon)
env.render()
env.close()

print(rewards)
plt.plot(running_mean(rewards,N))
plt.show()
