from os import O_NONBLOCK
import numpy as np
import pandas as pd

def get_condition_df(trial_df):
    trial_df = trial_df.copy()
    condition = []
    initial_state = trial_df.initial_state
    for i, it in enumerate(initial_state):
        # trial number, number of prey, x coor of prey, y coor of prey, x_vel of prey, occluder opacity
        condition.append([i, len(it['prey']), [np.round(prey.x, decimals=1) for prey in it['prey']], [np.round(prey.y, decimals=1) for prey in it['prey']], [np.round(prey.x_vel, decimals=3) for prey in it['prey']], it['occluders'][0].opacity == 255])
    condition_df = pd.DataFrame(data=condition, columns=['trial_num', 'n_prey', 'prey_x', 'prey_y', 'prey_x_vel', 'occluded'])    
    trial_df = trial_df.merge(condition_df, on='trial_num')
    return trial_df


def get_prey_pos_df(trial_df):
    prey_pos_all = []
    for (prey_pos, prey_x) in zip(trial_df.prey_pos, trial_df.prey_x) :
        prey_pos_dict = {np.round(x, 1): [] for x in prey_x}
        print(prey_pos_dict)
        for pp in prey_pos:
            for sprite in pp:
                prey_pos_dict[np.round(sprite[0], 1)].append(sprite)
        prey_pos_all.append(list(prey_pos_dict.values()))
    return prey_pos_all

def get_prey_end_step(trial_df):
    """Gets timepoint at which each prey goes past paddle y-position
    """
    prey_pos = trial_df.prey_pos
    prey_end_steps = []
    for pp in prey_pos: # each trial
        pes = []
        for ppp in pp: # each prey
            ppp = np.vstack(ppp)
            pes.append(np.where(ppp[:, 1] < (0.1 + 0.05))[0][0])
        prey_end_steps.append(pes)
    return prey_end_steps

def get_error(trial_df):
    prey_end_step = trial_df.prey_end_step
    prey_pos = trial_df.prey_pos
    agent_pos = trial_df.agent_pos
    error = []
    for t in range(len(trial_df)):
        pes = prey_end_step[t]
        pp = prey_pos[t]
        ap = agent_pos[t]
        final_pp = np.array([np.asarray(pp[prey])[pes[prey], 0] for prey in range(len(pp))])
        final_ap = np.array([np.asarray(ap)[prey_pes, 0] for prey_pes in pes])
        error.append(np.abs(final_ap - final_pp))
    return error

def get_success(trial_df):
    error = np.vstack(trial_df.error.to_numpy())
    success = np.asarray(error < 0.05, dtype=int)
    return success.tolist()
    
def get_prey_visible(trial_df):
    prey_pos = trial_df.prey_pos.to_numpy()
    occluders_pos = trial_df.occluders_pos
    occluders_pos = [op.reshape(-1, 2, 2) for op in occluders_pos]
    occluders_pos = [op[:, :, 0] for op in occluders_pos]
    occluders_pos = [(2 * op) + [1.1, -2.1] for op in occluders_pos]
    visible = []
    for (op, pp) in zip(occluders_pos, prey_pos):
        trial_visible = []
        for ppp in pp:
            ppp = np.array(ppp)
            top = op[:len(ppp), :]
            trial_visible.append(np.all((top[:, 0] <= ppp[:, 0], ppp[:, 0] <= top[:, 1], ppp[:, 1] <= 1), axis=0))
        visible.append(trial_visible)
    return visible

def get_prey_visible_step(trial_df):
    prey_visible = trial_df.prey_visible
    prey_visible_step = []
    for pv in prey_visible:
        trial_prey_visible_step = []
        for ppv in pv:
            ppvs = np.where(ppv)[0]            
            trial_prey_visible_step.append(ppvs[0] if len(ppvs) > 0 else -1)
        prey_visible_step.append(trial_prey_visible_step)
    return prey_visible_step
    

# def get_prey_visible_step(trial_df):
#     prey_visible = trial_df.prey_visible
#     prey_visible_step = [np.where(pv)[0] for pv in prey_visible]
#     prey_visible_step = [pvs[0] if len(pvs) > 0 else -1 for pvs in prey_visible_step]
#     return prey_visible_step


# def get_prey_visible(trial_df):
#     prey_pos = trial_df.prey_pos
#     prey_pos = [pp[:, 0] for pp in prey_pos]
#     occluders_pos = trial_df.occluders_pos
#     occluders_pos = [op.reshape(-1, 2, 2) for op in occluders_pos]
#     occluders_pos = [op[:, :, 0] for op in occluders_pos]
#     occluders_pos = [(2 * op) + [1.1, -2.1] for op in occluders_pos]
#     visible = [np.all((op[:, 0] <= pp, pp <= op[:, 1]), axis=0) for (pp, op) in zip(prey_pos, occluders_pos)]
#     return visible