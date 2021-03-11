import numpy as np
import tensorflow as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability import distributions as tfd
import argparse
from utils import *

tf1.compat.v1.enable_eager_execution()

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # IMPORTANT: out_size is still 2 in this case, because the action space is 2-dimensional. But your network will output some other size as it is outputing a distribution!
        # HINT 1: An example of this was given to you in Homework 1's Problem 1 in svm_tf.py. Now you will implement a multi-layer version.
        # HINT 2: You should use either of the following for weight initialization:
        #           - tf1.contrib.layers.xavier_initializer (this is what we tried)
        #           - tf.keras.initializers.GlorotUniform (supposedly equivalent to the previous one)
        #           - tf.keras.initializers.GlorotNormal
        #           - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        
        nn_output_size = out_size^2 + out_size
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(in_size,), name='x'),
                tf.keras.layers.Dense(16, activation = 'tanh', name = 'L1', kernel_initializer='glorot_normal', bias_initializer='zeros'),
                tf.keras.layers.Dense(nn_output_size, name = 'y_est', kernel_initializer='glorot_normal', bias_initializer='zeros')
            ]
        )
        
        self.model.summary()
        
        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (? x |O|) tensor that keeps a batch of observations
        # IMPORTANT: First two columns of the output tensor must correspond to the mean vector!

        y_est = self.model(x)
        
        return y_est
        
        ########## Your code ends here ##########


   
def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the negative log-likelihood loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: You may find the classes of tensorflow_probability.distributions (imported as tfd) useful.
    #       In particular, we used MultivariateNormalTriL, but it is not the only way.
    mu = y_est[:, 0:1]
    L = tf.convert_to_tensor([[y_est[:,2], y_est[:,3]], [y_est[:,4], y_est[:,5]]], dtype=tf.float32)
    eps = 0.0001
    sigma = tf.add(tf.matmul(L, tf.transpose(L)), eps * tf.eye(2, batch_shape=[L.shape[0]]))
    
    distributions = tfd.MultivariateNormalFullCovariance(loc = mu, covariance_matrix = sigma)
    log = tf.log(distributions.prob(y).eval())
    loss = -tf.reduce_mean(log)
    return loss
    
    ########## Your code ends here ##########


def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096*32,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')
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
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=1e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
