import tensorflow as tf
params =[]
o_n = tf.placeholder(dtype=tf.float32, shape=[None, 4])
params.append(o_n)
#imageIn = tf.reshape(o_n, shape=[-1, 84, 84, 1])
w = tf.truncated_normal([4,8], dtype=tf.float32)
params.append(w)
out = tf.matmul(o_n,w)
params.append(out)
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-3)

# Launch the graph in a session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(optimizer.get_slot_names())
# Evaluate the tensor `c`.
print(a)
print(sess.run(a))
feed_dict={o_n:[[1.,2.,2.,3.]]}
print(sess.run(o_n,feed_dict))
print(out)
print(sess.run(out,feed_dict))
#list = sess.run(params)
#print(list)
# from tensorflow.python import pywrap_tensorflow
#
# #checkpoint_path = os.path.join(model_dir, "model.ckpt")
# # Read data from checkpoint file
# reader = pywrap_tensorflow.NewCheckpointReader("./checkpoint_pnn_adam/col0/model-2800.cptk")
# var_to_shape_map = reader.get_variable_to_shape_map()
# # Print tensor name and values
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     print(reader.get_tensor(key).shape)
#     #print(reader.get_tensor(key))