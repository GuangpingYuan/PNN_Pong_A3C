# PNN_Pong_A3C
An implementation of Progressive Neural Network on Pong game using A3C structure.
Nowadays, applications of Machine Learning algorithms for multi-tasks are strongly limited by the strict data protection law and time-consuming data collection job. It is crucial to use the same data to solve as many similar tasks as possible. Humans and animals have the ability to continually acquire, fine-tune, and transfer knowledge and skills throughout their lifespan. This ability is referred to as lifelong learning or often called continual learning. Although state-of-the-art deep neural networks (DNNs) show powerful abilities to solve many complex tasks, most of the structures still suffer from catastrophic forgetting. The pre-trained model will be totally overwritten when it trains on a new task. Continual learning aims to solve this problem. The Progressive Neural Networks algorithm is one of the continual learning methods and has all abilities that continual learning has: Agents can not only acquire experience from a sequence of tasks but also have the ability to improve convergence speed.
Progressive Neural Network (PNN for short) starts from a single column: a normal deep convolutional neural network with L layers with an activation output h each. It is trained to solve the first task. Then, we add a second column with input 2 and it is supposed to deal with the second task. To speed up the convergence and transfer the knowledge from the first task, some lateral connections are utilized. The first layer is initialized randomly and the rest of the layers receives input from both columns via lateral connections. Each layer combines hidden activations of previous columns and a new column according to this formula. The third column is built the same way.
This work focuses on the application of PNN on deep reinforcement learning. Atari Pong game is used as an example.
I used the A3C model in my case. The Asynchronous Advantage Actor-Critic method combines the benefits of both value-based and policy-based model. 
I trained the A3C model with a CNN which has 2 convolutional layers and a fully connected layer.A fully conneced layer with linear output is the value output layer. Another fully connected layer with Softmax output is the policy output. The value is estimated to evaluate how good a certain state is and the policy is used to calculate the probability of actions. 
----------------------------------------------
How to use the code
----------------------------------------------
For training on the first task: 

1.Set the number of workers in run.py

2.Set the task number

"Pong_PNN.py --column=0" for training on the first task on the Linux

change "parser.add_argument('--column', type=int, default=0)" in main function

3.Set the name of environment

4.Create the environment in A3C.py in class Worker()

5.Set GPU device

6.Set num_workers to open multiple threads.

7.The checkpoint files will be save in the path'checkpoint_pnn_adam/col0', weights and biases are saved in .npy files.

8.If restore_from_save is set as 'True', the model will be trained from the last stop point.

For training on the second task:

1.Set the task number as 1

For testing in show.py

1.Set the ENV_NAME

2.Set the column_number

package version:

python3.6
tensorflow 1.12.0
atari-py 0.1.7
gym 0.12

Reference: Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R. & Hadsell, R. (2016) Progressive neural networks. arXiv at: http://arxiv.org/abs/1606.04671.
