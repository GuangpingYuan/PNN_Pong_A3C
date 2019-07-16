import gym
import argparse
import tensorflow as tf
import PNN
import Pong_PNN
import prog_nn
from Pong_Env import Pong_zoom
s_size = 7056
a_size = 6
checkpoint_list = [2800,3350,250] #'./checkpoint_pnn_adam/col0/checkpoint2800.npy'
checkpoint_path = './checkpoint_pnn_adam'
column_number = 0
def session():
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

    return session,global_episodes


def prog_model(input_shape,num_actions,session, prog_func_params,scope,reuse=False):
    checkpoint_base_path = './checkpoint_pnn_adam'
    kernel = [[8, 8, 1, 16], [4, 4, 16, 32], [3872, 256]]
    stride = [[1, 4, 4, 1], [1, 2, 2, 1], [1, 1, 1, 1]]
    activations = tf.nn.elu
    if column_number < 1:
        column = prog_nn.InitialColumnProgNN(s_size, kernel,stride,activations,session,checkpoint_base_path)
        #Two way to restore the model, from .npy file restore the weights
        #column.restore_weights(checkpoint_list[0])

    else:
        #Restore previous columns here
        prev_columns = []
        for i in range(column_number+1):
            print("reconstructing column i: ", i)
            if i==0:
                col_i = prog_nn.InitialColumnProgNN(s_size, kernel,stride,activations,session,checkpoint_base_path)
                #col_i.restore_weights(checkpoint_list[0])
                prev_columns.append(col_i)
            else:
                column = prog_nn.ExtensibleColumnProgNN(s_size,kernel,stride,activations,session,checkpoint_base_path,prev_columns)
                #column.restore_weights(checkpoint_list[i])
            print("column successfully restored")
        print("previous columns are:", prev_columns)
    return column


def main():
    #set the environment
    env = 'Pong-v0'
    #env = 'Pong_zoom'
    #env = 'Pong_noisy'
    model_path = '.\checkpoint_pnn_adam\col'+str(column_number)
    sess,global_episodes = session()
    master_network = PNN.AC_Network(s_size, a_size, 'global', None,prog_model, sess,
                                col_params= None)  # Generate global network
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-3)
    worker = PNN.Worker(env, 0, s_size, a_size, optimizer, checkpoint_path,
                              global_episodes, prog_model, sess, col_params=None)

    saver = tf.train.Saver(max_to_keep=5)
    if column_number == 0:
        #load the checkpoint of the first column
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        #load the final model
        #optimistic_restore(sess,'./checkpoint_pnn_adam/col1/model-3350.cptk')
        print('Loading the first column...')
    elif column_number ==1:
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        #load the final model to test on the second task
        #optimistic_restore(sess, './checkpoint_pnn_adam/col2/model-250.cptk')
        print('Loading the second column...')
    elif column_number ==2:
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        #optimistic_restore(sess,'./checkpoint_pnn_adam/col2/model-250.cptk')
        print('Loading the third column...')
    sess.run(global_episodes.initializer)

    for i in range (1000):
        worker.test(sess)

def optimistic_restore(session, save_file):
  """
  restore only those variable that exists in the model
  :param session:
  :param save_file:
  :return:
  """
  reader = tf.train.NewCheckpointReader(save_file)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted([(var.name, var.name.split(':')[0]) for
                      var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
  restore_vars = []
  name2var = dict(zip(map(lambda x: x.name.split(':')[0],tf.global_variables()),tf.global_variables()))
  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name]:
          #print("going to restore.var_name:",var_name,";saved_var_name:",saved_var_name)
          restore_vars.append(curr_var)
          #print(var_name)
      else:
          print("variable not trained.var_name:",var_name)
  saver = tf.train.Saver(restore_vars)
  saver.restore(session, save_file)


if __name__ == '__main__':
        main()