import os
import sys
current_dir = sys.path[0].replace("\\", "/")
project_dir = os.sep.join(current_dir.split('/')[:-2]).replace("\\", "/")
sys.path.append(project_dir)
import numpy as np
from argparse import ArgumentParser
from core.gail_agent import Gail
from core.diffusion_agent import DiffusionAgent

hyperparameters = \
    {
        'highway': {

                "gail": {'model_name': 'Gail', 'load_model_id': 0,'max_action': [np.pi/4, 5], 'state_dim': 56, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 1,'num_threads': 6 , 'use_running_state': False, 'save_dir': 'examples/model_data', 'expert_traj_path': 'data/ngsim/ngsim_clear_day_all_lc_trajs_v3.p', 'vehicle_dir': "data/ngsim/trajectory_set",},
                "diffusion": {'model_name': 'diffusion', 'load_model_id': 0, 'max_action': [np.pi/4, 5], 'state_dim': 56, 'action_dim': 2, 'num_epochs': 1, 'save_dir': 'examples/model_data', 'expert_traj_path': 'data/ngsim/ngsim_clear_day_all_lc_trajs_v3.p', 'vehicle_dir': "data/ngsim/trajectory_set",},
                },
        'intersection': {
                "gail": {'model_name': 'Gail', 'load_model_id': 0, 'max_action': [np.pi/4, 5], 'state_dim': 56, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 1, 'num_threads': 6 ,'use_running_state': False, 'save_dir': 'examples/model_data', 'expert_traj_path': 'data/interaction/interaction_all_trajs_v4.p', 'vehicle_dir': "data/interaction/trajectory_set",},
                "diffusion": {'model_name': 'diffusion', 'max_action': [np.pi / 4, 5], 'state_dim': 56, 'action_dim': 2, 'num_epochs': 1, 'save_dir': 'examples/model_data', 'expert_traj_path': 'data/interaction/interaction_all_trajs_v4.p', 'vehicle_dir': "data/interaction/trajectory_set"},
                },
    }


def main(args):
    scenario_name = args.scenario
    model = args.model
    param = hyperparameters[scenario_name][model]
    if args.num_threads is not None:
        param['num_threads'] = args.num_threads
    param['load_model_id'] = args.load_model_id

    if param['model_name'] == 'Gail':

        agent = Gail(args, param['state_dim'], param['action_dim'], param['max_action'], load_model_id=param['load_model_id'], max_iter_num=param['max_iter_num'], num_threads=param['num_threads'],
                     use_running_state=param['use_running_state'], save_dir=param['save_dir'], project_dir=project_dir,lr=param['lr'],
                     scenario=scenario_name, expert_traj_path=param['expert_traj_path'], vehicle_dir=param['vehicle_dir'], train=False)
        agent.exp(epochs=args.epochs, show=args.show, save_video=args.save_video)

    elif param['model_name'] == 'diffusion':

        agent = DiffusionAgent(args, param['state_dim'], param['action_dim'], param['max_action'], load_model_id=param['load_model_id'], num_epochs=param['num_epochs'], project_dir=project_dir,
                               scenario=scenario_name, save_dir=param['save_dir'], expert_traj_path=param['expert_traj_path'], vehicle_dir=param['vehicle_dir'], train=False)

        agent.exp(epochs=args.epochs, show=args.show, save_video=args.save_video)

def make_args():
    args = ArgumentParser()
    args.add_argument('--project-dir', default=project_dir, type=str,  help=""" It's one of two: highway and intersection.""")
    args.add_argument('--scenario', default='highway', type=str,  help=""" It's one of two: highway and intersection.""")
    args.add_argument('--model', default='diffusion', type=str, help=""" gail, diffusion, ppo, dqn""")
    args.add_argument('--load-model-id', default=900, type=int, help="""model epoch num""")
    args.add_argument('--update-mode', default='ppo', type=str, help=""" for ppo agent, ppo, ppo_v2.""")
    args.add_argument('--action-type', default='continuous', type=str, required=False, help=""" continue or discrete""")
    args.add_argument('--exp-name', default='exp_1', type=str)
    args.add_argument('--num-threads', default=1, type=int)
    args.add_argument('--num-epochs', default=1000, type=int)
    args.add_argument('--epochs', default=1000, type=int)
    args.add_argument('--seed', default=None, type=int)
    args.add_argument('--fps', default=5, type=int)
    args.add_argument('--show', action='store_true', help="render")
    args.add_argument('--save-video', action='store_true', help="save videos")
    args.add_argument('--upload', action='store_true', help="upload exp data to wandb server.")
    # for ddpg
    args.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    args.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    args.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    args.project_dir = project_dir
    return args


if __name__ == '__main__':
    args = make_args().parse_args()
    main(args)