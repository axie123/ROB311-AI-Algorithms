import gym
import numpy as np

from rl_agent import CartPoleAgent

# Test harness used to evaluate a CartPoleAgent.
SUCCESS_REWARD = 270
SUCCESS_STREAK = 100
MAX_EPISODES = 1500
MAX_STEPS = 5000

def run_cart_pole():
    """
    Run instances of cart-pole gym and tally scores.
    
    The function runs up to 1,000 episodes and returns when the 'success' 
    criterion for the OpenAI cart-pole task (v0) is met: an average reward
    of 195 or more over 100 consective episodes.
    """
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_STEPS

    # Create an instance of the agent.
    cp_agent = CartPoleAgent(env.observation_space, env.action_space)
    avg_reward, win_streak = (0, 0)
    rewards = []

    for episode in range(MAX_EPISODES):
        state = env.reset()

        # Reset the agent, if desired.
        cp_agent.reset()
        episode_reward = 0
    
        # The total number of steps is limited (to avoid long-running loops).
        for steps in range(MAX_STEPS):
            #env.render()

            # Ask the agent for the next action and step accordingly.
            action = cp_agent.action(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward

            # Update any information inside the agent, if desired.
            cp_agent.update(state, action, reward, state_next, terminal)
            episode_reward += reward # Total reward for this episode.
            state = state_next

            if terminal:
                # Update average reward.
                if episode < SUCCESS_STREAK:
                    rewards.append(episode_reward)
                    avg_reward = float(sum(rewards))/(episode + 1)
                else:
                    # Last set of epsiodes only (moving average)...
                    rewards.append(episode_reward)
                    rewards.pop(0)
                    avg_reward = float(sum(rewards))/SUCCESS_STREAK

                # Print some stats.
                print("Episode: " + str(episode) + \
                      ", Reward: " + str(episode_reward) + \
                      ", Avg. Reward: " + str(avg_reward))
  
                # Is the agent on a winning streak?
                if episode_reward >= SUCCESS_REWARD:
                    win_streak += 1
                else:
                    win_streak = 0
                break

        #print(rewards)

        # Has the agent succeeded?
        if win_streak == SUCCESS_STREAK and avg_reward >= SUCCESS_REWARD:
            return episode + 1, avg_reward 

    # Worst case, agent did not meet criterion, so bail out.
    return episode + 1, avg_reward

if __name__ == "__main__":
    episodes, achieved_avg_reward = run_cart_pole()
    print("--------------------------")
    print("Episodes to solve: " + str(episodes) + \
          ", Achieved Avg. Reward: " + str(achieved_avg_reward))
