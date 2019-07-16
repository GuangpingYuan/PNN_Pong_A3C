import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import pandas as pd
import skimage as skimage
from skimage import  color
import gym

from Pong_Env import Pong_noisy,Pong_flip,Pong_zoom

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes screen image to produce grey and resized image.
def process_frame(frame):
    s = skimage.color.rgb2gray(frame)
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer



result_path = './Results'
class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer,prog_model,session, col_params):
        with tf.variable_scope(scope):
            #Input and visual encoding layers


            self.out = prog_model(input_shape=s_size, num_actions=a_size,session=session,prog_func_params=col_params,
                                     scope='outparameter', reuse=False)
            self.policy = self.out.policy
            self.value =  self.out.value
            #self.value = tf.squeeze(self.value)
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 50.0)
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                session.run(tf.variables_initializer(trainer.variables()))
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


class Worker():
    def __init__(self, game, name, s_size, a_size,trainer, model_path, global_episodes,prog_model,session,col_params):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.data = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                 columns=["Step", "Reward", "Length", "Value", "Value Loss", "Policy Loss", "Entropy",
                                          "Grad Norm", "Var Norm"])
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer,prog_model,session,col_params)
        self.update_local_ops = update_target_graph('global', self.name)

        # Setup the environment
        if game == 'Pong-v0':
            self.env = gym.make('Pong-v0')
            print('create Pong environment')
        elif game == 'Pong_flip':
            self.env = Pong_flip()
            print('create Pong flip environment')
        elif game == 'Pong_noisy':
            self.env = Pong_noisy()
            print('create Pong noisy environment')
        elif game == 'Pong_zoom':
            self.env = Pong_zoom()
            print('create Pong zoom environment')
        self.actions = np.identity(a_size, dtype=bool).tolist()

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        feed_dict = self.local_AC.out.add_input_to_feed_dict(feed_dict,np.vstack(observations))

        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                s = self.env.reset()
                episode_frames.append(s)
                s = process_frame(s)


                while d == False:
                    #Take an action using probabilities from policy network output.
                    #Transfer observation to PNN
                    feed_dict = self.local_AC.out.add_input_to_feed_dict({},[s])
                    a_dist, v = self.local_AC.out.session.run([self.local_AC.policy,self.local_AC.value],feed_dict)
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    s1, r, d, info = self.env.step(a)

                    if d == False:
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v])
                    episode_values.append(v)

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 50 and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        feed_dict = self.local_AC.out.add_input_to_feed_dict({}, [s])
                        v1 = self.local_AC.out.session.run(self.local_AC.value, feed_dict)[0,0]

                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                print(self.number, episode_count, episode_reward)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 50 == 0 and self.name == 'worker_0':
                        self.local_AC.out.save(episode_count)
                        saver.save(sess,self.local_AC.out.save_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    self.data =self.data.append({"Step": episode_count, "Reward": float(mean_reward), "Length": float(mean_length),
                                      "Value": float(mean_value),
                                      "Value Loss": float(v_l), "Policy Loss": float(p_l), "Entropy": float(e_l),
                                      "Grad Norm": float(g_n), "Var Norm": float(v_n)}, ignore_index=True)
                    self.data.to_csv(result_path + '/result-' + str(self.number) + '.csv', sep=',', index=True)
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

    def test(self,sess):
        episode_count = sess.run(self.global_episodes)
        with sess.as_default(), sess.graph.as_default():
            episode_frames = []
            episode_reward = 0
            d = False

            s = self.env.reset()
            episode_frames.append(s)
            s = process_frame(s)

            while d == False:
                feed_dict = self.local_AC.out.add_input_to_feed_dict({},[s])
                a_dist, v = self.local_AC.out.session.run([self.local_AC.policy,self.local_AC.value],feed_dict)
                a = np.random.choice(a_dist[0],p=a_dist[0])
                a = np.argmax(a_dist == a)

                s1, r, d, info = self.env.step(a)
                self.env.render()
                if d == False:
                    episode_frames.append(s1)
                    s1 = process_frame(s1)
                else:
                    s1 = s

                episode_reward += r
                s = s1

            print(self.number, episode_count, episode_reward)
            sess.run(self.increment)





