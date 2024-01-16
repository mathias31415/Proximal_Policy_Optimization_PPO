import gymnasium as gym

# reward shaping for mountain car environment

class CustomMountainCarEnv_velocity(gym.Wrapper):             # Reward wird vergeben wenn velocity hoch ist
    def __init__(self, env):
        super(CustomMountainCarEnv_velocity, self).__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)    # "normale" step methode aufrufen

        #print("Velocity: ", observation[1])
        if observation[1] < -0.02 or observation[1] > 0.02:
            reward = 2

        if observation[1] < -0.03 or observation[1] > 0.03:
            reward = 4

        if observation[1] < -0.04 or observation[1] > 0.04:
            reward = 8

        if observation[1] < -0.05 or observation[1] > 0.05:
            reward = 16

        if observation[1] < -0.06 or observation[1] > 0.06:
            reward = 32

        if terminated:
            reward = 1000
            #print("######## Terminated ########")

        return observation, reward, terminated, truncated, info
    


class CustomMountainCarEnv_acceleration(gym.Wrapper):       # Reward wird vergeben wenn car sich in bewegungsrichtung beschleunigt
    # implementation works only for discrete action space
    def __init__(self, env):
        super(CustomMountainCarEnv_acceleration, self).__init__(env)

    def step(self, action):
        oldObs = self.state
        observation, reward, terminated, truncated, info = self.env.step(action)    # "normale" step methode aufrufen
               
        # newObs = self.state    # observation und newObs sind identisch, newObs hat gleiche Anzahl Nachkommastellen wie oldObs
        # if newObs[0] - oldObs[0] > 0 and action == 2: reward = 3 #moving right and pushing right = a reward
        # if newObs[0] - oldObs[0] < 0 and action == 0: reward = 3 #moving left and pushing left = a reward

        oldVelocity = oldObs[1]
        if oldVelocity > 0 and action == 2: 
            reward = 3 # moving right and pushing right = a reward
        if oldVelocity < 0 and action == 0: 
            reward = 3 # moving left and pushing left = a reward

        if terminated:
            reward = 1000
            #print("######## Terminated ########")
        
        return observation, reward, terminated, truncated, info
    
    

class CustomMountainCarEnv_position(gym.Wrapper):       # Reward wird vergeben je höher car kommt
    def __init__(self, env):
        super(CustomMountainCarEnv_position, self).__init__(env)

    def step(self, action):

        observation, reward, terminated, truncated, info = self.env.step(action)    # "normale" step methode aufrufen
 
        x_position = observation[0]
        if x_position >= -0.5:
            exp = int((x_position + 0.5) * 10)  # Exponent berechnen (z.B. -0.4 -> 0, -0.3 -> 1, ..., 0,4 -> 8)
            reward = 2 ** exp
            #print("Reward:", reward)

        # elif x_position >= -0.6:                
        #     exp = int((-x_position - 0.6) * 10)  # Exponent berechnen (z.B. -0,6 -> 0, -0,7 -> 1, ..., -1,2 -> 6)
        #     reward = 2 ** exp

        return observation, reward, terminated, truncated, info