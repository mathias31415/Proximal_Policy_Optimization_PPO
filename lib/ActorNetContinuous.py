# Description: This file contains the ActorNetContinuous class, which represents the stochastic-policy (policy = weights from the nn) for continuous action spaces.
import tensorflow as tf

# *************************************************MODIFIED************************************************** 
class ActorNetContinuous(tf.keras.Model):    # represents/ approximates the stochastic-policy (policy = weights from the nn)
    def __init__(self, units=(400, 300), n_actions=1, **kwargs):    # input = observation shape(batchsize, observation_shape) -> same as in discrete action space 
        super(ActorNetContinuous, self).__init__(**kwargs)
        self._layers = []
        n_outputs = n_actions*2 # one continuous output distribution contains values std and mean for gaussian 
        for i, u in enumerate(units):
            self._layers.append(tf.keras.layers.Dense(u, activation='relu'))
        self._layers.append(tf.keras.layers.Dense(n_outputs, activation = 'tanh'))   # output = ?? shape(batchsize, n_outputs)
        # modify output dimension to n_actions * 2 (= 1 for MountainCarCont) -> output is now std and mean of continuous gaussian distribution
        # modify output layer activation function -> use no activation/ linear activation a(x) = x to output the estimated values directly
        # if custom clipping is necessary, use tanh as output activation function to clip[-1,1]
        # in discrete action space 'softmax' exp(x) / tf.reduce_sum(exp(x)) calculates the value of each output vector in that way, the output can be interpreted as a discrete probability distribution (sum vectors = 1)
        
    # forward pass through the network
    def call(self, inputs, **kwargs):
        outputs = inputs
        for l in self._layers:
            outputs = l(outputs)
        # if last layer is reached, prepare the output to return mean and std
        mean, log_std = tf.split(outputs, 2, axis=-1)  # Split the output(Tensor shape(batchsize, n_outputs)) into 2 tensors (mean and log_std (ln!)) along the last axis(collums)
        #print('mean', mean,'log_std',log_std)
        return mean, log_std