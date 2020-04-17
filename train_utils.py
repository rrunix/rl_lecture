import custom_frozen_lake
import numpy as np
import gym
import seaborn as sns
import tabulate
import ipywidgets as widgets
from IPython import display
from tqdm.notebook import tqdm
import itertools
import numpy as np
import collections
import time
from cachier import cachier

gym.envs.registration.register(id="custom-FrozenLake-v0", entry_point="custom_frozen_lake:CustomFrozenLakeEnv")

env_params = {
    '@env': 'FrozenLake-v0',
    'is_slippery': {'values': [True, False], 'default': True, 'text': 'Resbaladizo?'}
}

custom_env_params = {
    '@env': 'custom-FrozenLake-v0',
    'prop_prob_action': {'values': (1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6), 'default': 1, 
                         'text': 'Proporción acción correcta', 'transformation': lambda x: 1 - (2 / (x + 2)), 'dest': 'prob_action'},
    'rg_hole':  {'values': (-3, -2, -1, -0.75, -0.5, -0.25, 0), 'default': 0, 'text': 'Recompensa agujero'},
    'rg_floor': {'values': (-0.5, -0.4, -0.3, -0.2, -0.1, 0), 'default': 0, 'text': 'Recompensa suelo'}
}

envs = {
    env_params['@env']: env_params,
    custom_env_params['@env']: custom_env_params,   
}

def get_policy_from_q(Q, n_states, n_actions):
    Q = Q.copy()
    p = np.zeros(shape=(n_states, n_actions))
    for state in range(n_states):
        if state in Q and Q[state].any():
            max_idx = np.argmax(Q[state])
            p[state, max_idx] = 1
        else:
            p[state, :] = 1 / n_actions
    return p


def display_lake_q(env, Q, ax=None):
    idxs, vals = zip(*[(k, Q[k]) for k in sorted(Q.keys())])
    vals = np.array(vals)
    return sns.heatmap(vals, xticklabels=['a'+str(i) for i in range(vals.shape[1])], yticklabels=['s'+str(i) for i in idxs])


def display_lake_value(env, v, ax=None):
    v = v.reshape(env.desc.shape)
    annot = np.round(np.copy(v), 4).astype(np.object)
    env_desc = env.desc.astype(str)
    annot[env_desc == 'G'] = 'G'
    annot[env_desc == 'H'] = 'H'
    return sns.heatmap(v, annot=annot, fmt='', ax=ax)
    

def display_lake_value_history(env, v_history):
    @widgets.interact(x=widgets.IntSlider(min=0, max=len(v_history)-1, step=1))
    def show(x):
        display_lake_value(env, v_history[x])
        
        
def display_lake_policy(env, policy, ax=None):
    action_map = {0: '←', 1: '↓', 2: '→', 3: '↑', -1: 'x'}
    env_desc = env.desc.astype(str)
    policy_repr = np.array([action_map[np.argmax(s)] for s in policy]).reshape(env_desc.shape)
    policy_repr[env_desc == 'G'] = 'G'
    policy_repr[env_desc == 'H'] = 'H'
    display.display_html(tabulate.tabulate(policy_repr, tablefmt='html'))
    

def display_lake_qpolicy(env, Q, ax=None):
    p = get_policy_from_q(Q, env.nS, env.action_space.n)
    display_lake_policy(env, p, ax=ax)
    
    
def multi_plot(*plots):
    def show(env, arg):
        for plot in plots:
            plot(env, arg)
    return show


def multi_display(*plots):
    def show(env, args):
        for arg, plot in zip(args, plots):
            if plot is not None:
                plot(env, arg)
                
    return show

                
def get_widget(desc):
    w =  widgets.SelectionSlider(
            options=desc['values'],
            value=desc['default'],
            description=desc['text'],
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            style = {'description_width': '100px'},
            readout=True)
    return w

    
def do_calculate_hash(args, kwargs):
    env_name, method_name, env_kwargs, kwargs = args
    return hash((env_name, method_name, frozenset(env_kwargs.items()), frozenset(kwargs.items())))


def cached_call(method, env_name, env_kwargs={}, method_kwargs={}):
    
    @cachier(hash_params=do_calculate_hash)
    def do_calculate(env_name, method_name, env_kwargs, kwargs):
        env_kwargs = env_kwargs.copy()
        kwargs = kwargs.copy()
        
        env = gym.make(env_name, **env_kwargs)
        env.reset()
        res = method(env, **kwargs), env
        env.close()
        return res
    
    return do_calculate(env_name, method.__name__, collections.OrderedDict(env_kwargs), collections.OrderedDict(method_kwargs))


def calculate(method, env_def, **kwargs):
    kwargs = kwargs.copy()
    
    env_kwargs = {}
    for k in env_def:
        if k in kwargs:
            arg = kwargs.pop(k)
            darg = env_def[k]

            if 'transformation' in darg:
                arg = darg['transformation'](arg)

            env_kwargs[darg.get('dest', k)] = arg
        
    return cached_call(method, env_def['@env'], env_kwargs, kwargs)
    
    
def display_learning_widget(method, method_params, display_f, env_params=env_params):
    selectors = {key:get_widget(desc) for key, desc in {**method_params, **env_params}.items() if key != '@env'}
    
    @widgets.interact(**selectors)
    def show(**kwargs):
        res, env = calculate(method, env_params, **kwargs)
        display_f(env, res)
        env.close()
        

def text_display(env):
    display.clear_output(True)
    display.display(env.render())
    

def play_episode(env, p, max_steps=100, delay=0.1, display_f=text_display, encode_state=None):
    state = env.reset()
    
    if encode_state:
        state = encode_state(state)
        
    reward = None
    for _ in range(max_steps):
        time.sleep(delay)
        display.clear_output(True)
        display.display(env.render())
        action = np.argmax(p[state])
        state, reward, done, _ = env.step(action)
        
        if done:
            break
            
        if encode_state:
            state = encode_state(state)

    
    print("Rewards {}".format(reward))
        