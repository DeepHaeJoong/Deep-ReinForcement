import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0

'''
while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done)
    reward_sum += reward
'''

'''
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()
'''

# Constants defining our neural network
learing_rate = 1e-1
input_size = env.observation_space.shape[0]  # 4 개
output_size = env.action_space.n  # left, right

X = tf.placeholder(tf.float32, [None, input_size], name = "input_x")  # 'None = 1'

# First layer of weights
W1 = tf.get_variable("W1", shape = [input_size, output_size], initializer = tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W1)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(shape = [None, output_size], dtype = tf.float32)

# loss function
loss = tf.reduce_sum(tf.square(Y - Qpred))

# Learning
train = tf.train.AdadeltaOptimizer(learning_rate = learing_rate).minimize(loss)

# Values for q learning

num_episodes = 2000
dis = 0.9
rList = []

# Setting up our environment
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(num_episodes):
    e = 1. / ((i / 10) + 1)
    rAll = 0
    step_count = 0
    s = env.reset()
    done = False

    # The Q-Network trainning
    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])
        # Choose an action by greedily (with a chane of random action) from the Q-network
        # The Q-network
        Qs = sess.run(Qpred, feed_dict = {X: x})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            # Choose an action by gredily from Q-network
            action = np.argmax(mainDQN.predict(state))

        # Save the experience to our buffer
        replay_buffer.append((state, action, reward, next_state, done))


        s1, reward, done, _ = env.step(a)
        if done: # Fail!
            Qs[0, a] = -100 # reward!
        else:
            x1 = np.reshape(s1, [1, input_size])
            # Obtain the Q' values by feeding the new state through our network
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
            Qs[0, a] = reward + dis * np.max(Qs1)

        # Train our network using target and predicted Q values
        sess.run(train, feed_dict={X: x, Y: Qs})
        s = s1

    rList.append(step_count)
    print("Episode:{} steps: {}".format(i, step_count))
    # if last 10's & avg steps are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

# See our traind network in action
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict = {X: x})
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum = reward
    if done:
        print("Total score : {}".format(reward_sum))
        break
