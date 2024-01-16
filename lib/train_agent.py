# training of the agent in the enviroment
import numpy as np
from lib.log_data import log_metrics
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf


def training_rollouts(env, agent, log_dir, epochs = 100, n_rollouts = 5, batch_size = 8, learn_steps = 16, render=False):
    summary_writer = tf.summary.create_file_writer(log_dir)     # create summary writer for tensorboard --> log_dir = path to log directory
    print('start training')
    total_timesteps = 0
    for epoch in range(epochs):    # one epoch -> one complete learning iteration bzw. one update of the frozen nets
        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        terminations = 0
        
        # collect experience in the enviroment with current policy for n episodes/ rollouts
        # give the agent more time to collect experiences more apart from the starting state
        for rollout in range(n_rollouts):
            obs, _ = env.reset()
            done = False 

            while not done:
                #env.render()    # call gui if render_mode = 'human'
                action = agent.act(np.array([obs]))
                new_obs, revard, termination, truncation, _ = env.step(action)
                if termination:
                    terminations += 1
                if termination or truncation:
                    done = True

                states.append(obs)
                rewards.append([revard])
                actions.append([action])
                obs = new_obs
                next_states.append(obs)
                dones.append([done])

                total_timesteps += 1

                if render:
                    clear_output(wait=True)
                    plt.axis('off')
                    plt.imshow(env.render())
                    plt.show()
                    print("done = ", done)
                    print("action = ", action)
                    print("total_timesteps = ", total_timesteps)
                    print("rollout = ", rollout)
                    print("epoch = ", epoch)
        
        # normalize states and next_states (observations) --> especially for hopper we need to deal with infinite observation spaces!
        # normalization should be consistent over the whole epoch            
        obs_mean = np.mean(states, axis=0, keepdims=True)
        obs_std = np.std(states, axis=0, keepdims=True)
        r_mean = np.mean(rewards, axis=0, keepdims=True)
        r_std = np.std(rewards, axis=0, keepdims=True)
        states = (states - obs_mean) / (obs_std + 1e-8) # add 1e-8 for numerical stability
        next_states = (next_states - obs_mean) / (obs_std + 1e-8)  # both normalized with the same mean and std -> common practice in machine learning
        rewards = (rewards - r_mean) / (r_std + 1e-8)
        
        # store colledted experience for all rollouts/ episodes and reset enviroment for next episode/ rollout
        states, actions, rewards, next_states, dones = map(np.array, [states, actions, rewards, next_states, dones])
        #obs, _ = env.reset() # TODO kann weg weil in der schleife oben schon resetet wird?
        print('collecting experience in rollouts finished, start learning phase')

        # learn policy and value from the collected data 
        for learn_step in range(learn_steps):
            indices = np.arange(states.shape[0])
            np.random.shuffle(indices)  # create random indice row
            
            # switch indices to random experience distribution
            shuffled_states = states[indices]
            shuffled_actions = actions[indices]
            shuffled_rewards = rewards[indices]
            shuffled_next_states = next_states[indices]
            shuffled_dones = dones[indices]

            # divides the whole shuffled experience into batches of batch_size
            for j in range(0, states.shape[0], batch_size):
                states_batch = shuffled_states[j:j + batch_size]    # j:j + batch_size -> returns all elements from x*batch_size to (x+1)*batch_size
                actions_batch = shuffled_actions[j:j + batch_size]
                rewards_batch = shuffled_rewards[j:j + batch_size]
                next_states_batch = shuffled_next_states[j:j + batch_size]
                dones_batch = shuffled_dones[j:j + batch_size]
                
                #print('try to call learn method with shuffled data')
                # push one batch of the shuffled experience to the learning method -> one update of the current nets (actor and critic) per passed batch of experience
                actor_loss, critic_loss = agent.learn(states_batch,
                                                    actions_batch,
                                                    rewards_batch,
                                                    next_states_batch,
                                                    dones_batch)
            print(f'update online nets, learn step {learn_step} of {learn_steps} finished --> actor loss {actor_loss}, critic loss {critic_loss}')


        agent.update_frozen_nets()
        print(f'update frozen nets, epoche {epoch} of {epochs} finished')

        avg_epoch_return = np.sum(rewards)/n_rollouts
        sum_epoch_terminations = terminations
        
        #do some more prints for analyzing the model behavior while training
        print(f'===> epoch {epoch + 1}, total_timesteps {total_timesteps}, actor loss {actor_loss}, critic loss {critic_loss}, avg_epoch_return {avg_epoch_return}, sum_epoch_terminations {sum_epoch_terminations}')
        
        # Log metrics at the end of each epoch to tensorboard
        log_metrics(summary_writer = summary_writer,
                    epoch = epoch, 
                    terminations  = terminations,
                    total_timesteps = total_timesteps,
                    critic_loss = critic_loss, 
                    actor_loss= actor_loss,
                    avg_epoch_return = avg_epoch_return,
                    actor_learning_rate = agent.actor_learning_rate, 
                    critic_learning_rate = agent.critic_learning_rate,
                    epsilon = agent.epsilon, 
                    gamma = agent.gamma)
        
            
    env.close() # kill gui



def training_steps(env, agent, log_dir, epochs = 100, n_steps = 2048, batch_size = 8, learn_steps = 16, render=False):
    summary_writer = tf.summary.create_file_writer(log_dir)     # create summary writer for tensorboard --> log_dir = path to log directory
    print('start training')
    total_timesteps = 0
    for epoch in range(epochs):    # one epoch -> one complete learning iteration bzw. one update of the frozen nets
        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        terminations = 0
        steps = 0
        rollout = 0
        
        # collect experience in the enviroment with current policy for n episodes/ rollouts
        # give the agent more time to collect experiences more apart from the starting state
        while steps < n_steps:
            obs, _ = env.reset()
            done = False 
            rollout += 1
            #print('start new rollout')

            while not done:
                #env.render()    # call gui if render_mode = 'human'
                action = agent.act(np.array([obs]))
                noise = np.random.normal(0, scale = 0.005, size=action.shape)
                noisy_action = action + noise
                new_obs, revard, termination, truncation, _ = env.step(action)
                if termination:
                    terminations += 1
                if termination or truncation:
                    done = True

                states.append(obs)
                rewards.append([revard])
                actions.append([action])
                obs = new_obs
                next_states.append(obs)
                dones.append([done])

                total_timesteps += 1
                steps += 1

                if render:
                    clear_output(wait=True)
                    plt.axis('off')
                    plt.imshow(env.render())
                    plt.show()
                    print("done = ", done)
                    print("action = ", action)
                    print("reward = ", revard)
                    print("total_timesteps = ", total_timesteps)
                    print("rollout = ", rollout)
                    print("epoch = ", epoch)
        
        # normalize states and next_states (observations) --> especially for hopper we need to deal with infinite observation spaces!
        # normalization should be consistent over the whole epoch            
        obs_mean = np.mean(states, axis=0, keepdims=True)
        obs_std = np.std(states, axis=0, keepdims=True)
        states = (states - obs_mean) / (obs_std + 1e-8) # add 1e-8 for numerical stability
        next_states = (next_states - obs_mean) / (obs_std + 1e-8)  # both normalized with the same mean and std -> common practice in machine learning

        # store colledted experience for all rollouts/ episodes and reset enviroment for next episode/ rollout
        states, actions, rewards, next_states, dones = map(np.array, [states, actions, rewards, next_states, dones])
        #obs, _ = env.reset() # TODO kann weg weil in der schleife oben schon resetet wird?
        print('collecting experience in rollouts finished, start learning phase')

        # learn policy and value from the collected data 
        for learn_step in range(learn_steps):
            indices = np.arange(states.shape[0])
            np.random.shuffle(indices)  # create random indice row
            
            # switch indices to random experience distribution
            shuffled_states = states[indices]
            shuffled_actions = actions[indices]
            shuffled_rewards = rewards[indices]
            shuffled_next_states = next_states[indices]
            shuffled_dones = dones[indices]

            # divides the whole shuffled experience into batches of batch_size
            for j in range(0, states.shape[0], batch_size):
                states_batch = shuffled_states[j:j + batch_size]    # j:j + batch_size -> returns all elements from x*batch_size to (x+1)*batch_size
                actions_batch = shuffled_actions[j:j + batch_size]
                rewards_batch = shuffled_rewards[j:j + batch_size]
                next_states_batch = shuffled_next_states[j:j + batch_size]
                dones_batch = shuffled_dones[j:j + batch_size]
                
                #print('try to call learn method with shuffled data')
                # push one batch of the shuffled experience to the learning method -> one update of the current nets (actor and critic) per passed batch of experience
                actor_loss, critic_loss = agent.learn(states_batch,
                                                    actions_batch,
                                                    rewards_batch,
                                                    next_states_batch,
                                                    dones_batch)
            print(f'update online nets, learn step {learn_step} of {learn_steps} finished --> actor loss {actor_loss}, critic loss {critic_loss}')

        agent.update_frozen_nets()
        print(f'update frozen nets, epoche {epoch} of {epochs} finished')

        cumm_epoch_return = np.sum(rewards)
        sum_epoch_terminations = terminations
        
        #do some more prints for analyzing the model behavior while training
        print(f'===> epoch {epoch + 1}, total_timesteps {total_timesteps}, actor loss {actor_loss}, critic loss {critic_loss}, cumm_epoch_return {cumm_epoch_return}, sum_epoch_terminations {sum_epoch_terminations}')
        
        # # render the learned policy
        # obs, _ = env.reset()
        # done = False 
        # _return = 0

        # while not done:
        #     #env.render()    # call gui if render_mode = 'human'
        #     action = agent.act(np.array([obs]))
        #     new_obs, revard, termination, truncation, _ = env.step(action)

        #     obs = new_obs
        #     _return += revard

        #     if render:
        #         clear_output(wait=True)
        #         plt.axis('off')
        #         plt.imshow(env.render())
        #         plt.show()
        #     if termination or truncation:
        #         done = True
        # print("return = ", _return)

        # Log metrics at the end of each epoch to tensorboard
        log_metrics(summary_writer = summary_writer,
                    epoch = epoch, 
                    terminations  = terminations,
                    total_timesteps = total_timesteps,
                    critic_loss = critic_loss, 
                    actor_loss= actor_loss,
                    avg_epoch_return = cumm_epoch_return,
                    actor_learning_rate = agent.actor_learning_rate, 
                    critic_learning_rate = agent.critic_learning_rate,
                    epsilon = agent.epsilon, 
                    gamma = agent.gamma)
        
            
    env.close() # kill gui