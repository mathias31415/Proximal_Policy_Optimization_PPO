# Critic Network for continous and discrete action spaces
import tensorflow as tf

class CriticNet(tf.keras.Model):   # evaluates choosen actions(critic) in reference to the estimated actions(target critic) -> provides feedback to the actor (optipizing the policy was better/ worser)
    def __init__(self, units=(64, 64), **kwargs):   # # change network units to MLPolicy from SB3 (64,64)
        super(CriticNet, self).__init__(**kwargs)
        self._layers = []
        for i, u in enumerate(units):
            self._layers.append(tf.keras.layers.Dense(u, activation='relu'))
        self._layers.append(tf.keras.layers.Dense(1))
        
    def call(self, inputs, **kwargs):
        outputs = inputs
        for l in self._layers:
            outputs = l(outputs)
        return outputs