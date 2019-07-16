import numpy
import tensorflow as tf
import gym
import argparse
import os
import time
import threading

import multiprocessing
from Pong_Env import Pong_noisy,Pong_flip,Pong_zoom
import PNN
import prog_nn
COL_NUM = 'column_number'
CHECKPOINT_PATH = 'checkpoint_pnn_adam'
CHECKPOINT_LIST ='checkpoint_list'
restore_from_save = 'restore_from_save'
RESTORE_CHECKPOINT = 'restore_from_save_path'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
max_episode_length = 1000
gamma = 0.99
s_size = 7056
a_size = 6
#num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
num_workers = 8
def get_env_parameter():
    env_0 = {
        'name':'Pong-v0',
        'env': gym.make('Pong-v0')
    }
    env_1 = {
        'name':'Pong_zoom',
        'env' :Pong_zoom()
    }
    env_2 = {
        'name':'Pong_noisy',
        'env': Pong_noisy()
    }
    env_3 = {
        'name':'Pong_flip',
        'env' :Pong_flip()}
    return [env_0,env_1,env_2,env_3]

def get_col_parameter():
    #For column number 0
    col_0_params = {
        COL_NUM: 0,
        CHECKPOINT_PATH: 'checkpoint_pnn_adam',
        CHECKPOINT_LIST: [],
        restore_from_save: False,
        RESTORE_CHECKPOINT: CHECKPOINT_PATH + '/col0'
    }
    #For column number 1
    col_1_params = {
        COL_NUM: 1,
        CHECKPOINT_PATH: 'checkpoint_pnn_adam',
        CHECKPOINT_LIST: [2800], #Change to latest checkpoint for col 0
        restore_from_save: False,
        RESTORE_CHECKPOINT: CHECKPOINT_PATH + '/col1'
    }
    #For column number3
    col_2_params = {
        COL_NUM: 2,
        CHECKPOINT_PATH: 'checkpoint_pnn_adam',
        CHECKPOINT_LIST: [2800,3350],
        restore_from_save: False,
        RESTORE_CHECKPOINT: CHECKPOINT_PATH + '/col2'
    }
    #For column number3
    col_3_params = {
        COL_NUM: 3,
        CHECKPOINT_PATH: 'checkpoint_pnn_adam',
        CHECKPOINT_LIST: [2800,3350,500],
        restore_from_save: False,
        RESTORE_CHECKPOINT: CHECKPOINT_PATH + '/col3'
    }
    #For column number4
    col_4_params = {
        COL_NUM: 4,
        CHECKPOINT_PATH: 'checkpoint_pnn_adam',
        CHECKPOINT_LIST: [],
        restore_from_save: False
    }
    return [col_0_params,col_1_params,col_2_params,col_3_params,col_4_params]

def get_session(args):
    env_para = get_env_parameter()
    env = env_para[args.column]['name']
    tf.reset_default_graph()


    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)

    workers, col_parameter, global_episodes = get_threads(env,num_workers,session,args)
    saver = tf.train.Saver(max_to_keep=5)
    coord = tf.train.Coordinator()
    return session, coord, saver, workers, col_parameter, global_episodes


def get_threads(env,num_workers,session,args):
    with tf.device("/gpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        session.run(global_episodes.initializer)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-3)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=.99)
        col_parameter = get_col_parameter()
        master_network = PNN.AC_Network(s_size, a_size, 'global', None, prog_model, session,
                                        col_params=col_parameter[args.column])  # Generate global network
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(PNN.Worker(env, i, s_size, a_size, optimizer,col_parameter[args.column][CHECKPOINT_PATH],
                                      global_episodes, prog_model, session, col_params=col_parameter[args.column]))


        return workers, col_parameter, global_episodes


def pong_train(session, coord,a3c_params,max_episode_length, gamma,saver,workers):
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma,session, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        time.sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

def prog_model(input_shape,num_actions,session, prog_func_params,scope,reuse=False):
    column_number = prog_func_params[COL_NUM]
    checkpoint_base_path = prog_func_params[CHECKPOINT_PATH]
    checkpoint_list = prog_func_params[CHECKPOINT_LIST]
    if column_number < 1:
        kernel =[[8,8,1,16],[4,4,16,32],[3872,256]]
        stride = [[1, 4, 4, 1], [1, 2, 2, 1], [1, 1, 1, 1]]
        activations = tf.nn.elu

        column = prog_nn.InitialColumnProgNN(s_size, kernel,stride,activations,session,checkpoint_base_path)
        #column.restore_weights(checkpoint_list[0])

    else:
        #Restore previous columns here
        prev_columns = []
        for i in range(column_number):
            #topology =[1,16,32,3872,256,6]
            kernel = [[8,8,1,16],[4,4,16,32],[3872,256]]
            stride = [[1, 4, 4, 1], [1, 2, 2, 1], [1, 1, 1, 1]]
            activations = tf.nn.elu
            print("reconstructing column i: ", i)
            if i==0:

                col_i = prog_nn.InitialColumnProgNN(s_size, kernel,stride,activations,session,checkpoint_base_path)
            else:
                col_i = prog_nn.ExtensibleColumnProgNN(s_size,kernel,stride,activations,session,checkpoint_base_path,prev_columns)

            col_i.restore_weights(checkpoint_list[i])
            prev_columns.append(col_i)
            print("column successfully restored")
        print("previous columns are:", prev_columns)

        column = prog_nn.ExtensibleColumnProgNN(s_size, kernel,stride,activations,session,checkpoint_base_path,prev_columns)
    return column

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--column', type=int, default=0)
    args = parser.parse_args()
    assert(args.column >= 0 and args.column <5)
    session, coord,saver,workers, col_parameter, global_episodes = get_session(args)
    if col_parameter[args.column][restore_from_save] == True:
        print('Loading Model...')
        model_path = col_parameter[args.column][RESTORE_CHECKPOINT]
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    elif col_parameter[args.column]['column_number'] ==0:
        session.run(tf.global_variables_initializer())
        print('Initialized')
    else:
        pass
        #optimistic_restore(session,'./checkpoint_pnn_adam/col0/model-2800.cptk')

    #print(session.run(tf.report_uninitialized_variables()))
    pong_train(session, coord,col_parameter[args.column],max_episode_length,gamma,saver,workers)


if __name__ == '__main__':
        main()
