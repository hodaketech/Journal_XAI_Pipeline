import matplotlib.pyplot as plt
from cliffworld_env import CliffWorldEnv
from agents import QLearningAgent, drQAgent, HRAAgent

EPISODES = 1000
env = CliffWorldEnv()
state_size = env.observation_space.n
action_size = env.action_space.n

agents = {
    'QLearningAgent': QLearningAgent(state_size, action_size),
    'drQAgent': drQAgent(state_size, action_size),
    'HRAAgent': HRAAgent(state_size, action_size),
}

rewards = {name: [] for name in agents}

for episode in range(EPISODES):
    for name, agent in agents.items():
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
        rewards[name].append(total_reward)
        agent.epsilon = max(agent.epsilon * 0.995, 0.05)

# Plot results
for name in agents:
    plt.plot(rewards[name], label=name)
plt.ylim([-120, 0])
plt.title('CliffWorld-v0 Agent Comparison')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cliffworld_comparison_final.png", dpi=300)
plt.show()
