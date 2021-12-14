import os
import pandas as pd
from moog import sprite as sprite_lib
from matplotlib import path as mpl_path
from matplotlib import transforms as mpl_transforms
import matplotlib.pyplot as plt
import collections
import json
import numpy as np
import seaborn as sns
import colorsys
from utils.log import log_shape, log_dtypes

_SPRITE_LOG_INDEX = 5
_PREY_HUES = [0.1666, 0.3333, 0.6667, 0.1667, 0.5, 0.8333]
_AGENT_HUE = [0.5]
ATTRIBUTES_FULL = list(sprite_lib.Sprite.FACTOR_NAMES)

ATTRIBUTES_PARTIAL = [
    'x', 'y', 'x_vel', 'y_vel', 'angle', 'id']

ATTRIBUTES_PARTIAL_INDICES = {k: i for i, k in enumerate(ATTRIBUTES_PARTIAL)}

def _check_cols(cols, df):
    for col in cols:
        if col not in df.columns: 
            raise KeyError('%s not in dataframe' % col)

def create_new_sprite(sprite_kwargs, vertices=None):
    """Create new sprite from factors.
    Args:
        sprite_kwargs: Dict. Keyword arguments for sprite_lib.Sprite.__init__().
            All of the strings in sprite_lib.Sprite.FACTOR_NAMES must be keys of
            sprite_kwargs.
        vertices: Optional numpy array of vertices. If provided, are used to
            define the shape of the sprite. Otherwise, sprite_kwargs['shape'] is
            used.
    Returns:
        Instance of sprite_lib.Sprite.
    """
    if vertices is not None:
        # Have vertices, so must invert the translation, rotation, and
        # scaling transformations to get the original sprite shape.
        center_translate = mpl_transforms.Affine2D().translate(
            -sprite_kwargs['x'], -sprite_kwargs['y'])
        x_y_scale = 1. / np.array([
            sprite_kwargs['scale'],
            sprite_kwargs['scale'] * sprite_kwargs['aspect_ratio']
        ])
        transform = (
            center_translate +
            mpl_transforms.Affine2D().rotate(-sprite_kwargs['angle']) +
            mpl_transforms.Affine2D().scale(*x_y_scale)
        )
        vertices = mpl_path.Path(vertices)
        vertices = transform.transform_path(vertices).vertices

        sprite_kwargs['shape'] = vertices

    return sprite_lib.Sprite(**sprite_kwargs)

def get_trial_paths(base_path):
    dataset_paths = [f.path for f in os.scandir(base_path)]
    dataset_trial_paths = []
    for dataset_path in dataset_paths:  
        trial_paths = [
            os.path.join(dataset_path, x)
            for x in sorted(os.listdir(dataset_path)) if x.isnumeric()
        ]
        dataset_trial_paths += trial_paths
    return dataset_trial_paths

def attributes_to_sprite(a):
    """Create sprite with given attributes."""
    attributes = {x: a[i] for i, x in enumerate(ATTRIBUTES_FULL)}

    if len(a) > len(ATTRIBUTES_FULL):
        vertices = np.array(a[-1]) 
    else:
        vertices = None
    # print(attributes, len(a), len(ATTRIBUTES_FULL), a[-1], vertices)
    return create_new_sprite(attributes, vertices=vertices)

def get_condition_df(trial_df):
    trial_df = trial_df.copy()
    condition = []
    initial_state = trial_df.initial_state
    for i, it in enumerate(initial_state):
        condition.append([i, np.round(it['prey'][0].x, decimals=1), np.round(it['prey'][0].x_vel, decimals=3), it['occluders'][0].opacity == 255])
    condition_df = pd.DataFrame(data=condition, columns=['trial_num', 'prey_x', 'prey_vel', 'occluded'])    
    trial_df = trial_df.merge(condition_df, on='trial_num')
    return trial_df

def get_states(trial, sprites_list):
    def _attributes_to_sprite_list(sprite_list):
        return [np.array(s)[np.asarray((0, 1, -1))] for s in sprite_list]

    states = [collections.OrderedDict([
        (k, _attributes_to_sprite_list(v))
        for k, v in trial[i+2][_SPRITE_LOG_INDEX] if k in sprites_list # TODO: Remove this
    ]) for i in range(len(trial)-2)]
    
    return states

def get_prey_pos_from_states(trial_df):
    prey_pos = []
    for state in trial_df.states:
        prey_pos_dict = {}
        for step in state:
            prey_state = step['prey']
            for p in prey_state: # for each prey
                if p[-1] not in prey_pos_dict:
                    prey_pos_dict[p[-1]] = []
                prey_pos_dict[p[-1]].append(p[:2])
        prey_pos.append(list(prey_pos_dict.values()))
    return prey_pos

            

def get_initial_state(trial):
    """Get initial state OrderedDict."""
    def _attributes_to_sprite_list(sprite_list):
        return [attributes_to_sprite(s) for s in sprite_list]

    state = collections.OrderedDict([
        (k, _attributes_to_sprite_list(v))
        for k, v in trial[0][_SPRITE_LOG_INDEX] if k != 'walls' # TODO: Remove this
    ])
    
    return state

def get_states_df(trial_df, sprites_list=['agent', 'prey']):
    states_df = trial_df.copy()
    states = []
    for trial_path in trial_df.trial_path:
        trial = json.load(open(trial_path, 'r'))
        trial_states = get_states(trial, sprites_list)
        states.append(trial_states)
    states_df['states'] = states
    return states_df

def get_initial_state_df(trial_df):
    initial_state_df = trial_df.copy()
    initial_states = []
    for trial_path in trial_df.trial_path:
        trial = json.load(open(trial_path, 'r'))
        initial_state = get_initial_state(trial)
        initial_states.append(initial_state)
    initial_state_df['initial_state'] = initial_states
    return initial_state_df


def get_trial_df(trial_paths):    
    """Create trial dataframe with features for each trial stimulus.
    This dataframe has one row per trial.
    """

    trial_df = pd.DataFrame({
        'trial_num': range(len(trial_paths)),
        'trial_path': trial_paths
    })

    # stim_feature_keys = set()
    # for stim_f in stimulus_features:
    #     stim_feature_keys.update(stim_f.keys())
    # stim_feature_keys = list(stim_feature_keys)
    
    # for column_name in stim_feature_keys:
    #     column = [stim_f.get(column_name, None) for stim_f in stimulus_features]
    #     trial_df[column_name] = column

    return trial_df

def _get_sprite_pos(sprite_string, step_string):
    x_ind = ATTRIBUTES_PARTIAL_INDICES['x']
    y_ind = ATTRIBUTES_PARTIAL_INDICES['y']

    for x in step_string[-1]:
        if x[0] == sprite_string:
            sprite = [[x_attr[x_ind], x_attr[y_ind]] for x_attr in x[1]]
            return sprite


def _get_prey_pos(step_string):
    return _get_sprite_pos('prey', step_string)

def _get_agent_pos(step_string):
    return _get_sprite_pos('agent', step_string)

def _get_occluders_pos(step_string):
    return _get_sprite_pos('occluders', step_string) 

def get_prey_pos(trial, sample_every=1):
    step_indices = np.arange(0, len(trial) - 2, sample_every)
    prey_pos = []
    for step in step_indices:
        step_string = trial[step + 2]    
        prey_pos.append(_get_prey_pos(step_string))
    prey_pos = [np.array(x) for x in prey_pos if len(x) > 0 ]
    return prey_pos

def get_agent_pos(trial, sample_every=1):
    step_indices = np.arange(0, len(trial) - 2, sample_every)
    agent_pos = []
    for step in step_indices:
        step_string = trial[step + 2]
        agent_pos.append(_get_agent_pos(step_string))
    agent_pos = np.vstack([np.array(x) for x in agent_pos if len(x) > 0 ])
    return agent_pos

def get_occluders_pos(trial, sample_every=1):
    step_indices = np.arange(0, len(trial) - 2, sample_every)
    occluders_pos = []
    for step in step_indices:
        step_string = trial[step + 2]
        occluders_pos.append(_get_occluders_pos(step_string))
    occluders_pos = np.vstack([np.hstack(np.array(x)) for x in occluders_pos if len(x) > 0 ])
    occluders_pos = occluders_pos.reshape(-1, 2, 2)[:, :, 0] + [(1.1 + 0.35)/2, -1 * (1.1 + 0.35)/2]
    return occluders_pos

def get_success(trial_df):
    prey_pos = trial_df.prey_pos
    agent_pos = trial_df.agent_pos
    return np.vstack([int(np.abs((ap[-1, 0] - pp[-1, 0])) < 0.05) for (pp, ap) in zip(prey_pos, agent_pos)])

def get_agent_vel(trial_df):
    agent_pos = trial_df.agent_pos
    return [np.diff(ap[:, 0]) for ap in agent_pos]

def get_trial_end_step(trial_df):
    prey_pos = trial_df.prey_pos
    prey_pos = [np.vstack(pp) for pp in prey_pos]
    return np.vstack([np.where(pp[:, 1]  < 0.1 + 0.04)[0][0] for pp in prey_pos])

def get_prey_visible(trial_df):
    prey_pos = trial_df.prey_pos
    prey_pos = [pp[:, :] for pp in prey_pos]
    occluders_pos = trial_df.occluders_pos
    occluders_pos = [op.reshape(-1, 2, 2) for op in occluders_pos]
    occluders_pos = [op[:, :, 0] for op in occluders_pos]
    occluders_pos = [(2 * op) + [1.1, -2.1] for op in occluders_pos]
    visible = [np.all((op[:, 0] <= pp[:, 0], pp[:, 0] <= op[:, 1], pp[:, 1] <= 1.0), axis=0) for (pp, op) in zip(prey_pos, occluders_pos)]
    return visible

@log_shape
@log_dtypes
def get_occluders_pos_df(trial_df):
    occluders_pos_df = trial_df.copy()
    occluders_poss = []
    for trial_path in trial_df.trial_path:
        trial = json.load(open(trial_path, 'r'))
        occluders_pos = get_occluders_pos(trial)
        occluders_poss.append(occluders_pos)
    occluders_pos_df['occluders_pos'] = occluders_poss
    return occluders_pos_df

@log_shape
@log_dtypes
def get_prey_pos_df(trial_df):
    prey_pos_df = trial_df.copy()
    prey_poss = []
    for trial_path in trial_df.trial_path:
        trial = json.load(open(trial_path, 'r'))
        prey_pos = get_prey_pos(trial)
        prey_poss.append(prey_pos)
    prey_pos_df['prey_pos'] = prey_poss
    return prey_pos_df

@log_shape
@log_dtypes
def get_agent_pos_df(trial_df):
    agent_pos_df = trial_df.copy()
    agent_poss = []
    for trial_path in trial_df.trial_path:
        trial = json.load(open(trial_path, 'r'))
        agent_pos = get_agent_pos(trial)
        agent_poss.append(agent_pos)
    agent_pos_df['agent_pos'] = agent_poss
    return agent_pos_df

def trim_pos(trial_df, pos_col):
    trial_df = trial_df.copy()
    trial_end_step = trial_df.trial_end_step.to_numpy()
    pos = trial_df[pos_col].to_numpy()
    pos = [np.vstack(pp) for pp in pos]
    pos = [pp[:tes, :] for (tes, pp) in zip(trial_end_step, pos)]
    trial_df[pos_col] = pos
    return trial_df

def get_error(trial_df):
    agent_pos = trial_df.agent_pos.to_numpy()    
    agent_pos = [ap[-1, 0] for ap in agent_pos]
    prey_pos = trial_df.prey_pos.to_numpy()
    prey_pos = [pp[-1, 0] for pp in prey_pos]
    return [np.abs((ap - pp)) for (pp, ap) in zip(prey_pos, agent_pos)]

def get_prey_visible_step(trial_df):
    prey_visible = trial_df.prey_visible
    prey_visible_step = [np.where(pv)[0] for pv in prey_visible]
    prey_visible_step = [pvs[0] if len(pvs) > 0 else -1 for pvs in prey_visible_step]
    return prey_visible_step


def _get_trajectories(group):    
    agent_pos = group['agent_pos']    
    trial_length = int(np.min([len(l) for l in agent_pos]))
    stack_agent_pos = np.hstack([ap[-trial_length:] for ap in agent_pos])
    return stack_agent_pos[:, :, 0]

def get_trajectories_df(pos_df):
    _check_cols(('prey_vel', 'prey_x', 'occluded', 'agent_pos', 'prey_pos', 'prey_visible_step'), pos_df)
    pos_df = pos_df.copy()
    c_df = pos_df.groupby(['prey_vel', 'prey_x', 'occluded']).apply(lambda x: pd.Series({'agent_pos': _get_trajectories(x)}))
    return c_df

## PLOTTING FUNCTIONS
def display_condition_error(error_df):
    _check_cols(('error', 'prey_vel', 'prey_x', 'trial_num', 'occluded'), error_df)
    ce_df = error_df.groupby(['prey_vel', 'prey_x', 'occluded'])['error'].apply(lambda e: np.mean(e)[0]).reset_index(level=-1).pivot(columns='occluded')
    ce_df = ce_df.rename(columns={False: "occluded", True: "visible"}, level=1)['error']
    f, ax = plt.subplots(1,1, dpi=100)
    sns.scatterplot(data=ce_df, x='visible', y='occluded', ax=ax)
    ax.axline((-.2,-.2), (0.1, 0.1))


def display_condition_distr(condition_df):
    if 'prey_vel' not in condition_df.columns or 'prey_x' not in condition_df.columns or 'occluded' not in condition_df.columns:
        raise KeyError('prey_vel, prey_x, or occluded not in dataframe')
    f, axs = plt.subplots(1, 2)
    sns.histplot(data=condition_df, x='prey_vel', hue='occluded', ax=axs[0])
    sns.histplot(data=condition_df, x='prey_x', hue='occluded', ax=axs[1])
    plt.show()
    condition_df = condition_df.copy()
    condition_df.occluded = np.int8(condition_df.occluded)
    sns.pairplot(condition_df[['prey_x', 'prey_vel', 'occluded']], kind='hist', plot_kws={'cbar': True})
    
def display_prey_agent_pos(pos_df, show_visible=True, sample_every=2):
    if 'prey_pos' not in pos_df.columns or 'occluders_pos' not in pos_df.columns or 'agent_pos' not in pos_df.columns:
        raise KeyError('prey_pos, agent_pos, or occluders_pos not in dataframe')
    if show_visible and 'prey_visible' not in pos_df.columns:
        raise KeyError('prey_visible not in dataframe')
    for _, trial in pos_df.iterrows():
        pp = trial.prey_pos[::sample_every]
        ap = trial.agent_pos[::sample_every]
        if show_visible:
            pv = np.clip(trial.prey_visible[::sample_every], 0.1, 1)
        else:
            pv = np.ones(len(pp))
        n_prey = len(pp[0]) # number of prey
        f, ax = plt.subplots(1, 1, dpi=100)        
        color = np.linspace(1., 0.666, len(ap))
        color = np.stack([colorsys.hsv_to_rgb(c, 1, 1) for c in color], axis=0)
        for i, (pos, prey_vis) in enumerate(zip(pp, pv)):            
            for p in range(n_prey):
                ax.scatter(pos[p][0], pos[p][1], color=color[i],  marker='.', alpha=prey_vis, s=15)
        for i, pos in enumerate(ap):
            ax.scatter(pos[0][0], pos[0][1], color=color[i],  marker='.', alpha=prey_vis)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([.05, 1.01])
            

