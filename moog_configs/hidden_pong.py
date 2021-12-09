"""Pong task.

This task is based on the task in the following paper:
Rajalingham, Rishi and Piccato, Aida and Jazayeri, Mehrdad (2021). The role of
mental simulation in primate physical inference abilities.

In this task the subject controls a paddle at the bottom of the screen with a
joystick. The paddle is constrained to only move left-right. Each trial one ball
falls from the top of the screen, starting at a random position and moving with
a random angle. The ball bounces off of vertical walls on either side of the
screen as it falls. The subject's goal is the intercept the ball with the
paddle.
"""

import collections
import numpy as np

from moog import action_spaces
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import sprite
from moog import tasks
from moog.state_initialization import distributions as distribs


def _get_config(occluder_shift_candidates, x_vel_candidates):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################


    # Prey 
    prey_factors = distribs.Product(
        [distribs.Discrete('x', candidates=np.linspace(0.1, 0.8, 8)),
         distribs.Discrete('x_vel', candidates=x_vel_candidates)],
        y=1.5, y_vel=-0.02, shape='circle', scale=0.07, c0=0.2, c1=1., c2=1.,
    )


    # Walls
    left_wall = [[0.05, -0.2], [0.05, 2], [-1, 2], [-1, -0.2]]
    right_wall = [[0.95, -0.2], [0.95, 2], [2, 2], [2, -0.2]]


    def state_initializer():
        walls = [
            sprite.Sprite(shape=np.array(v), x=0, y=0, c0=0., c1=0., c2=0.5)
            for v in [left_wall, right_wall]
        ]
        agent = sprite.Sprite(
            x=0.5, y=0.1, shape='square', aspect_ratio=0.2, scale=0.1, c0=0.33,
            c1=1., c2=0.66)

        # Occluders
        
        # DECENTERED OCCLUDER
        occluder_diam = 0.3
        occluder_rad = occluder_diam / 2
        occluder_center = np.random.choice(occluder_shift_candidates)
        occluder_shape_left = np.array([
            [-1.1, -0.1], [occluder_center - occluder_rad, -0.1], [occluder_center - occluder_rad, 1.1], [-1.1, 1.1]
        ])
        occluder_shape_right = np.array([
            [occluder_center + occluder_rad, -0.1], [2.1, -0.1], [2.1, 1.1], [occluder_center +  occluder_rad, 1.1]
        ])
        opacity = np.random.choice([120, 255], p=[0.3, 0.7])
 
        occluders = [sprite.Sprite(
            x=0., y=0., shape=occluder_shape_left, scale=1., c0=0., c1=0., c2=0.5, opacity=opacity), 
                sprite.Sprite(
            x=0., y=0., shape=occluder_shape_right, scale=1., c0=0., c1=0., c2=0.5, opacity=opacity), 
        ]

        window_offset = 0.1

        window_shape = np.array([
            [occluder_center - occluder_rad + window_offset, -0.1], [occluder_center + occluder_rad - window_offset, -0.1],
            [occluder_center + occluder_rad - window_offset, 1.1], [occluder_center - occluder_rad + window_offset, 1.1]])

        window = sprite.Sprite(x=0., y=0., shape=window_shape, c0=0.2, c1=1., c2=1., scale=1., opacity=0)

        # SHIFTED OCCLUDER        
        # occluder_rad = 0.35

        # occluder_shift = np.random.choice(occluder_shift_candidates)
        
        # occluder_shape_left = np.array([
        #     [-1.1, -0.1], [occluder_rad - occluder_shift, -0.1], [occluder_rad - occluder_shift, 1.1], [-1.1, 1.1]
        # ])
        # occluder_shape_right = np.array([
        #     [1 - occluder_rad - occluder_shift, -0.1], [2.1, -0.1], [2.1, 1.1], [1 - occluder_rad - occluder_shift, 1.1]
        # ])

        # opacity = np.random.choice([120, 255], p=[0.3, 0.7])
 
        # occluders = [sprite.Sprite(
        #     x=0., y=0., shape=occluder_shape_left, scale=1., c0=0., c1=0., c2=0.5, opacity=opacity), 
        #         sprite.Sprite(
        #     x=0., y=0., shape=occluder_shape_right, scale=1., c0=0., c1=0., c2=0.5, opacity=opacity), 
        # ]

        # window_offset = 0.1

        # window_shape = np.array([
        #     [occluder_rad - occluder_shift + window_offset, -0.1], [1 - occluder_rad - occluder_shift - window_offset, -0.1],
        #     [1 - occluder_rad - occluder_shift - window_offset, 1.1], [occluder_rad - occluder_shift + window_offset, 1.1]])

        # window = sprite.Sprite(x=0., y=0., shape=window_shape, c0=0.2, c1=1., c2=1., scale=1., opacity=0)


        state = collections.OrderedDict([            
            ('prey', [sprite.Sprite(**prey_factors.sample())]),
            ('occluders', occluders),
            ('window', [window]),
            ('agent', [agent]),
            ('walls', walls),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)
    asymmetric_collision = physics_lib.Collision(
        elasticity=1., symmetric=False, update_angle_vel=False)

    physics = physics_lib.Physics(
        (agent_friction_force, ['agent', 'occluders', 'window']),
        (asymmetric_collision, 'prey', 'walls'),
        updates_per_env_step=10,
    )

    ############################################################################
    # Task
    ############################################################################

    contact_task = tasks.ContactReward(1., layers_0='agent', layers_1='prey')
    reset_task = tasks.Reset(
        condition=lambda state: all([s.y < 0. for s in state['prey']]),
        steps_after_condition=15,
    )
    task = tasks.CompositeTask(contact_task, reset_task)

    ############################################################################
    # Action space
    ############################################################################

    # trying to make composite action spaces; not working
    # action_space_agent = action_spaces.Joystick(
    #     scaling_factor=0.005, action_layers='agent', constrained_lr=True)

    # action_space_occluder = action_spaces.Joystick(
    #     scaling_factor=0.005, action_layers='occluders', constrained_lr=True)

    # action_space_window = action_spaces.Joystick(
    #     scaling_factor=0.005, action_layers='window', constrained_lr=True)
    # action_space = action_spaces.Composite(agent=action_space_agent,  window=action_space_window) #occluder=action_space_occluder)
    
    action_space = action_spaces.Grid(
        scaling_factor=0.015,
        action_layers=['agent', 'window', 'occluders'],
        control_velocity=True,
        momentum=0.5,  
    )

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(64, 64), anti_aliasing=1, color_to_rgb='hsv_to_rgb')

    ############################################################################
    # Game rules
    ############################################################################

    prey_vanish = game_rules.VanishOnContact(
        vanishing_layer='prey',
        contacting_layer='agent',
    )

    def prey_contacting_window(state):
        if len(state['prey']) > 0:
            return state['window'][0].overlaps_sprite(state['prey'][0]) and state['occluders'][0].opacity > 0
        return False

    def _hide_occluders(sprite):
        sprite.opacity = 0

    hide_occluders = game_rules.ModifySprites(
        layers='occluders', modifier=_hide_occluders, sample_one=False,)

    hide_occluders = game_rules.ConditionalRule(
        condition=lambda s: prey_contacting_window(s),
        rules=hide_occluders,
    )
    # WITH HIDE OCCLUDERS RULES
    # rules = (prey_vanish, hide_occluders)

    # WITHOUT HIDE OCCLUDERS RULES
    rules = (prey_vanish, )
    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
        'game_rules': rules,
    }
    return config


def get_config(level):
    """Get config dictionary of kwargs for environment constructor.
    
    Args:
        level: Int. Different values yield different maze sizes and numbers of
            ghosts.
    """
    if level == 0:
        return _get_config(
            occluder_shift_candidates=[0],
            x_vel_candidates=[0],
        )
    elif level == 1:
        # nonzero x-velocity
        return _get_config(
            occluder_shift_candidates=[0],
            x_vel_candidates=[-0.01, -0.005, 0.005, 0.01],
        )
    elif level == 2:
        # shifted occluders
        return _get_config(
            occluder_shift_candidates=[0.2, 0.5, 0.8],
            x_vel_candidates=[0],
        )
    elif level == 3:
        # shifted occluders and nonzero x-velocity
        return _get_config(
            occluder_shift_candidates=[-0.3, -0.1, 0, 0.1, 0.3],
            x_vel_candidates=[-0.01, -0.005, 0.005, 0.01],
        )
    else:
        raise ValueError('Invalid level {}'.format(level))