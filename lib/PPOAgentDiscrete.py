# Description: Implementation of the PPO Agent for discrete action spaces
import tensorflow as tf
import numpy as np
from lib.ActorNetDiscrete import ActorNetDiscrete as Actor
from lib.CriticNet import CriticNet as Critic

class PPOAgentDiscrete:
    def __init__(self, action_space, observation_space, gamma=0.99, epsilon = 0.1, actor_learning_rate=0.00025, critic_learning_rate=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = action_space
        self.observation_shape = observation_space.shape[0]
        self.actor = Actor(n_actions=action_space.n)
        self.actor_old = Actor(n_actions=action_space.n)
        self.critic = Critic()
        self.target_critic = Critic()
        
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_learning_rate)   # default = 0,001 -> hatten wir auch schon
        self._init_networks()
        
    def _init_networks(self):
        initializer = np.zeros([1, self.observation_shape])   # ergänzt zu V1 -> hatten wir aber auch schon gemacht
        self.actor(initializer)
        self.actor_old(initializer)
        
        self.critic(initializer)
        self.target_critic(initializer)
        
        self.update_frozen_nets()
        
    def act(self, observation):
        probs = self.actor(observation).numpy()
        probs = np.squeeze(probs)
        action = np.random.choice(self.action_space.n, p=probs)
        return action
    
    def get_critic_grads(self, states, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            next_value = self.target_critic(next_states)
            q_value = rewards + (1-dones) * self.gamma * next_value
            value = self.critic(states)
            
            advantage = q_value - value
            loss = tf.reduce_mean(tf.square(advantage))
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        return gradients, loss, advantage
    
    def get_actor_grads(self, states, actions, advantage):
        with tf.GradientTape() as tape:
            p_current = tf.gather(self.actor(states), actions, axis=1)
            p_old = tf.gather(self.actor_old(states), actions, axis=1)
            ratio = p_current / p_old
            clip_ratio = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)
            # entropy loss hatten wir probiert, bringt aber wenig --> sollte eigentlich exploration förndern
            # standardize advantage
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.keras.backend.std(advantage) + 1e-8)
            objective = ratio * advantage
            clip_objective = clip_ratio * advantage
            loss = -tf.reduce_mean(tf.where(objective < clip_objective, objective, clip_objective))
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        return gradients, loss
        

    def learn(self, states, actions, rewards, next_states, dones):
        critic_grads, critic_loss, advantage = self.get_critic_grads(states, rewards, next_states, dones)
        actor_grads, actor_loss = self.get_actor_grads(states, actions, advantage)
        
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss, critic_loss
    
    def update_frozen_nets(self):
        # TODO: set discount factor  -> was soll der hier bringen?
        weights = self.actor.get_weights()
        self.actor_old.set_weights(weights)
        
        weights = self.critic.get_weights()
        self.target_critic.set_weights(weights)

    def save_models(self, actor_path='actor_weights.h5', critic_path='critic_weights.h5'):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_models(self, actor_path='actor_weights.h5', critic_path='critic_weights.h5'):
        try:
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            print('Model loaded sucessful')
        except Exception as e:
            print(f"Error: {e}")