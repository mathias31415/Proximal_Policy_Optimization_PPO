# Description: Implementation of the PPO Agent for continous action spaces
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from lib.ActorNetContinuous import ActorNetContinuous as Actor
from lib.CriticNet import CriticNet as Critic

class PPOAgentContinuous:
    def __init__(self, action_space, observation_space, gamma=0.99, epsilon = 0.1, actor_learning_rate=0.00025, critic_learning_rate=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.observation_shape = observation_space.shape[0]
# *************************************************MODIFIED**************************************************     
        self.action_shape = action_space.shape[0]
        self.actor = Actor(n_actions=self.action_shape)
        self.actor_old = Actor(n_actions=self.action_shape)
# ***********************************************************************************************************     

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

 # *************************************************MODIFIED**************************************************     
    def act(self, observation):
        mean_tensor, log_std_tensor = self.actor(observation) # Actor network will output the gaussian probability distribution of actions for the given observation-state
        mean = tf.gather(mean_tensor, indices= 0, axis=1).numpy()
        std = tf.gather(tf.exp(log_std_tensor), indices= 0, axis=1).numpy()
        #print('mean', mean, 'std', std)
        action = np.random.normal(size = self.action_shape, loc = mean, scale = std)  # modify sampling method to sample random actions in respect a continous normal distribution
        #print('Choosen Action:', action)
        return action
        # in continous action space this output should be a real number (optional clipped [-1,1])
        # for mnt car = force applied on the car (clipped [-1,1]) and * power od 0.0015 -> no custom action clipping necessary
 # ***********************************************************************************************************     
   
    def get_critic_grads(self, states, rewards, next_states, dones):    # parameters are adjusted to minimize the difference between predicted values and observed returns
        with tf.GradientTape() as tape:
            next_value = self.target_critic(next_states)
            q_value = rewards + (1-dones) * self.gamma * next_value
            value = self.critic(states)
            
            advantage = q_value - value
            loss = tf.reduce_mean(tf.square(advantage))
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        return gradients, loss, advantage
    
 # *************************************************MODIFIED**************************************************     
    def get_actor_grads(self, states, actions, advantage):  # parameters are updated to maximize the expected cumulative reward, incorporating feedback from the critic
        with tf.GradientTape() as tape:
            # get distribution from current policy (used to sample/ explore in enviroment)
            means_current, log_stds_current = self.actor(states)    # mean and log tensors shape(batchsize,1) wit batchsize = len(states)
            normal_dist_current = tfp.distributions.Normal(loc=means_current, scale=tf.exp(log_stds_current))
            # sample made actions from the approximated current distribution -> introduces more exploration in the action space by sampling specific actions rather than the whole distribution (mean and std)
            p_current = normal_dist_current.prob(actions)
            
            # get distribution from old policy (used to evaluate current policy in ratio)
            means_old, log_stds_old = self.actor_old(states)    # mean and log tensors shape(batchsize,1) wit batchsize = len(states)
            normal_dist_old = tfp.distributions.Normal(loc=means_old, scale=tf.exp(log_stds_old))
            # sample random actions from the approximated current distribution -> introduces more exploration in the action space by sampling actions rather than selecting the mean directly
            p_old = normal_dist_old.prob(actions)

            # calculate the ratio to weight the advantage estimate from the critic network (value-based)
            ratio = p_current / (p_old  + 1e-8)   #p_current, p_old, ratio tensors of shape(batchsize, batchsize, 1) -> like discrete implementation
            # + 1e-8 to avoid division by zero
# ***********************************************************************************************************     

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
        # TODO: set discount factor  -> not necessary
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
