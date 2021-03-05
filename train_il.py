import numpy as np
import argparse
import tensorflow as tf1
import tensorflow.compat.v2 as tf
from utils import *

tf1.compat.v1.enable_eager_execution()

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT 1: An example of this was given to you in Homework 1's Problem 1 in svm_tf.py. Now you will implement a multi-layer version.
        # HINT 2: You should use either of the following for weight initialization:
        #           - tf1.contrib.layers.xavier_initializer (this is what we tried)
        #           - tf.keras.initializers.GlorotUniform (supposedly equivalent to the previous one)
        #           - tf.keras.initializers.GlorotNormal
        #           - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        # initializer = tf1.contrib.layers.xavier_initializer()
        
        # shape = (in_size,32) #change this to actual shape
        # initializer = tf.keras.initializers.GlorotUniform()
        # self.W0 = tf.Variable(initializer(shape=shape))
        # self.b0 = tf.Variable(tf.zeros([32,]))
        # shape = (32, out_size)
        # self.W1 = tf.Variable(initializer(shape=shape))
        # self.b1 = tf.Variable(tf.zeros([out_size,]))
        
        # x_input = tf.keras.Input(shape=(in_size, 1), name='x')
        # conv1 = tf.keras.layers.Conv2D(32, (3,3), activation = 'tanh')(x_input)
        # conv2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'tanh')(conv1)
        # conv2 = tf.keras.layers.Flatten()(conv2)
        # y_est = tf.keras.layers.Dense((outsize, dim), activation = 'tanh', name='y_est')(conv2)
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(in_size,), name='x'),
                tf.keras.layers.Dense(32, activation = 'tanh', name = 'L1', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                tf.keras.layers.Dense(32, activation = 'tanh', name = 'L2', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                tf.keras.layers.Dense(out_size, name = 'y_est', kernel_initializer='glorot_uniform', bias_initializer='zeros')
            ]
        )
        
        self.model.summary()
        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (? x |O|) tensor that keeps a batch of observations
        y_est = self.model(x)

        return y_est

        ########## Your code ends here ##########


def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally
    steering_error = y_est[:,0] - y[:,0]
    throttle_error = y_est[:,1] - y[:,1]
    
    loss = tf.add(3*tf.norm(steering_error), 1*tf.norm(throttle_error))
    loss = tf.math.reduce_mean(loss)
    return loss

    ########## Your code ends here ##########
    

def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_IL')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        # HINT: You did the exact same thing in Homework 1! It is just the networks weights and biases that are different.
        
        with tf.GradientTape() as tape:
            y_est = nn_model.call(x)
            current_loss = loss(y_est, y)
        
        dl_dW = tape.gradient(current_loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(dl_dW, nn_model.trainable_variables))

        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_IL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
