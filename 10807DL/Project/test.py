import gym
env = gym.make('CartPole-v0')
env.monitor.start('tmp/cartpole-experiment-1', force=True)
env.reset()
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

env.monitor.close()