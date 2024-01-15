# Description: This file contains the ActorNetContinuous class, which represents the stochastic-policy (policy = weights from the nn) for continuous action spaces.
#              This implementation is modified after analyzing the performance advantages from stable baselines 3
import tensorflow as tf

class ActorNetContinuous(tf.keras.Model):    # represents/ approximates the stochastic-policy (policy = weights from the nn)
    def __init__(self, units=(64, 64), n_actions=None, **kwargs):    # change network units to MLPolicy from SB3 (64,64)
        super(ActorNetContinuous, self).__init__(**kwargs)
        self._hiddenlayers = []

        # Define the trainable log standard deviation (modified, instead of outputting the std by the network)
        # We expect this approach provides more flexibility in shaping the exploration strategy
        # -> we expect to need more exploration, because the behavoir/ policy of the agent becomes to deterministic
        # high std -> more exploration, low std -> more exploitation 
        # trainable variables are updated with every call of the learn method
        self.log_std = tf.Variable(tf.zeros([1, n_actions], dtype=tf.float32), trainable=True)

        for i, u in enumerate(units):
            self._hiddenlayers.append(tf.keras.layers.Dense(u, activation= tf.nn.leaky_relu)) # eyplanation for leaky_relu see below
        self._mean = tf.keras.layers.Dense(n_actions, activation = 'tanh')  
        
    # forward pass through the network
    def call(self, inputs, **kwargs):
        outputs = inputs
        for l in self._hiddenlayers:
            outputs = l(outputs)
        # if last layer is reached, prepare the output to return mean and std
        mean = self._mean(outputs)
        log_std = self.log_std
        return mean, log_std

