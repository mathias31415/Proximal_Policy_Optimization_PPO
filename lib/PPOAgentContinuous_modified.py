'''
Description: Implementation of the PPO Agent for continous action spaces

Changes from discrete implementation to continuous marked by: *****************
Changes to improve performance after analyzing the differences between this and the SB3 approach marked by: -----------------------

'''


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from lib.ActorNetContinuous_modified import ActorNetContinuous as Actor
from lib.CriticNet_modified import CriticNet as Critic

class PPOAgentContinuous:
    def __init__(self, action_space, observation_space, gamma=0.99, epsilon = 0.1, actor_learning_rate=0.00025, critic_learning_rate=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.observation_shape = observation_space.shape[0]
# *************************************************MODIFIED**************************************************     
        self.action_space = action_space
        self.action_shape = action_space.shape[0]
        self.actor = Actor(n_actions=self.action_shape)
        self.actor_old = Actor(n_actions=self.action_shape)
        print('action_shape bzw. Actor Outputs',self.action_shape)
# ***********************************************************************************************************     

        self.critic = Critic()
        self.target_critic = Critic()
        
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_learning_rate)   # default = 0,001 -> hatten wir auch schon
        self._init_networks()
        
    def _init_networks(self):
        initializer = np.zeros([1, self.observation_shape])
        self.actor(initializer)
        self.actor_old(initializer)
        
        self.critic(initializer)
        self.target_critic(initializer)
        
        self.update_frozen_nets()

 # *************************************************MODIFIED****************************************************************
 # -------------------Changed in second Improvement Cycle to tf notation (consistency to get_actor_grads)-------------------
    def act(self, observation):
        # experiment with tensorflow variables (consistency!)
        means_current, log_stds_current = self.actor(observation)    # mean and log tensors shape(batchsize,1) wit batchsize = len(states)
        normal_dist = tfp.distributions.Normal(loc=means_current, scale=tf.exp(log_stds_current))
        action_tf = normal_dist.sample()    # this introduces exploration to the agent -> sampling from stochastic policy!
        action = action_tf.numpy()[0]   # reduce shape (1,3) to (3,)
        clip_action = np.clip(action, self.action_space.low, self.action_space.high)    #action clipped to the limits of enviroments action space
        return clip_action
 # *************************************************************************************************************************  
 # -------------------------------------------------------------------------------------------------------------------------     
   
    def get_critic_grads(self, states, rewards, next_states, dones):    # parameters are adjusted to minimize the difference between predicted values and observed returns
        with tf.GradientTape() as tape:
            next_value = self.target_critic(next_states)    # like standard implementation
            value = self.critic(states)
            advantage = 0

 # ----------------Changed loss definition after SB3 analysis in second improvement cycle ----------------------------
            gae_lambda = 0.99 # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            next_non_terminal = (1-dones)

            # calculate new advantage definition from SB3 with TD-learning and additional coefficients
            delta = rewards + self.gamma * next_value * next_non_terminal - value   # = q-value - value
            advantage = delta + self.gamma + gae_lambda * next_non_terminal * advantage
            loss = tf.reduce_mean(tf.square(advantage))
# ---------------------------------------------------------------------------------------------------------------------

        # those grads not used in baselines, but changes not implemented here. -> Potential for further improvements
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]  
        return gradients, loss, advantage

    
 # *************************************************MODIFIED**************************************************     
    def get_actor_grads(self, states, actions, advantage):  # parameters are updated to maximize the expected cumulative reward, incorporating feedback from the critic
        with tf.GradientTape() as tape:
            # do calculations with tf-tensors because backprob is needed (consistency in used framework)
            # get distribution from current policy (used to sample/ explore in enviroment)
            means_current, log_stds_current = self.actor(states)    # mean and log tensors shape(batchsize,1) wit batchsize = len(states)
            normal_dist_current = tfp.distributions.Normal(loc=means_current, scale=tf.exp(log_stds_current))
            p_current = normal_dist_current.log_prob(actions)
            
            # get distribution from old policy (used to evaluate current policy in ratio)
            means_old, log_stds_old = self.actor_old(states)    # mean and log tensors shape(batchsize,1) wit batchsize = len(states)
            normal_dist_old = tfp.distributions.Normal(loc=means_old, scale=tf.exp(log_stds_old))
            # sample random actions from the approximated current distribution -> introduces more exploration in the action space by sampling actions rather than selecting the mean directly
            p_old = normal_dist_old.log_prob(actions)

            # calculate the ratio to weight the advantage estimate from the critic network (value-based)
            ratio = tf.exp(p_current - p_old)
# ***********************************************************************************************************     

            clip_ratio = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)

            # standardize advantage
            advantage_standard = (advantage - tf.reduce_mean(advantage)) / (tf.keras.backend.std(advantage) + 1e-8)
            objective = ratio * advantage_standard
            clip_objective = clip_ratio * advantage_standard
            policy_loss = -tf.reduce_mean(tf.where(objective < clip_objective, objective, clip_objective))

# ------------------Changed loss definition after SB3 analysis in second improvement cycle-----------------------
            ent_coef = 0.002295 # coefficients like BaselinesZoo by DLR
            vf_coef = 0.835671
            entropy_loss = -tf.reduce_mean(-p_current)
            value_loss = tf.reduce_mean(advantage)
            loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
# ---------------------------------------------------------------------------------------------------------------

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        return gradients, loss
        

    def learn(self, states, actions, rewards, next_states, dones):
        # same optimizer for critic and actor like baselines not implemented. 
        critic_grads, critic_loss, advantage = self.get_critic_grads(states, rewards, next_states, dones)
        actor_grads, actor_loss = self.get_actor_grads(states, actions, advantage)

        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        #print('##Actor Weights', self.actor.get_weights())

        return actor_loss, critic_loss
    
    def update_frozen_nets(self):   # no frozen nets in SB3 approach
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
            print('Model reloaded sucessful')
        except Exception as e:
            print(f"Error: {e}")
