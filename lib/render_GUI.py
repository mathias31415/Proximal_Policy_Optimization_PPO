# Function to render an agent in a specific gym enviroment in external GUI -> not so laggy like matplotlib implementation
import numpy as np

def render_GUI_GYM(render_env, render_agent):

    # Reset the enviroment and get the initial observation
    render_obs,_ = render_env.reset()

    # Start rendering in pyglet GUI (internal gym method which uses pyglet inthe background)
    render_env.render()
    episode = 0

    #Run the GUI until Keyboard Interrupt hits (only cell-interrupt in Jupyter-Notebook is supported. It's not possible to close the GUI directly!)
    try:
        while True:
            render_action = render_agent.act(np.array([render_obs]))
            render_obs, _, termination, truncation, _ = render_env.step(render_action)

            if termination or truncation:
                print(f'Episode {episode} finished')
                episode += 1
                render_obs,_ = render_env.reset()

    except KeyboardInterrupt as e:
        print('Closed Rendering sucessful')
        render_env.close()



def render_GUI_SB3(render_env, render_agent):

    # Reset the enviroment and get the initial observation
    render_obs = render_env.reset()

    # Start rendering in pyglet GUI (internal gym method which uses pyglet inthe background)
    render_env.render()
    episode = 0

    #Run the GUI until Keyboard Interrupt hits (only cell-interrupt in Jupyter-Notebook is supported. It's not possible to close the GUI directly!)

    try:    # for vec_enviroments
        while True:

            action, _states = render_agent.predict(obs)
            obs, rewards, dones, info = render_env.step(action)
            render_env.render("human")

            if dones:
                print(f'Episode {episode} finished')
                episode += 1
                render_obs  = render_env.reset()

    except KeyboardInterrupt as e:
        print('Closed Rendering sucessful')
        render_env.close()




def render_GUI(render_env, render_agent):

    # Reset the enviroment and get the initial observation
    render_obs,_ = render_env.reset()

    # Start rendering in pyglet GUI (internal gym method which uses pyglet inthe background)
    render_env.render()
    episode = 0

    #Run the GUI until Keyboard Interrupt hits (only cell-interrupt in Jupyter-Notebook is supported. It's not possible to close the GUI directly!)
    try:
        while True:
            render_action = render_agent.act(np.array([render_obs]))
            render_obs, _, termination, truncation, _ = render_env.step(render_action)

            if termination or truncation:
                print(f'Episode {episode} finished')
                episode += 1
                render_obs,_ = render_env.reset()

    except KeyboardInterrupt as e:
        print('Closed Rendering sucessful')
        render_env.close()