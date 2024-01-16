# Description: Actor network for discrete action space
import tensorflow as tf

class ActorNetDiscrete(tf.keras.Model):
    def __init__(self, units=(400, 300), n_actions=2, **kwargs):
        super(ActorNetDiscrete, self).__init__(**kwargs)
        self._layers = []
        for i, u in enumerate(units):
            self._layers.append(tf.keras.layers.Dense(u, activation='relu'))
        self._layers.append(tf.keras.layers.Dense(n_actions, activation='softmax'))
        
    def call(self, inputs, **kwargs):
        outputs = inputs
        for l in self._layers:
            outputs = l(outputs)
        return outputs