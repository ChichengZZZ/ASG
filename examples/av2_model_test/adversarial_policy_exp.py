import os
import sys
current_dir = sys.path[0].replace("\\", "/")
project_dir = os.sep.join(current_dir.split('/')[:-2]).replace("\\", "/")
sys.path.append(project_dir)
import numpy as np
from argparse import ArgumentParser
from core.natural_adversarial_agent import NaturalAdversarialAgent
from core.gail_agent import Gail
from core.diffusion_agent import DiffusionAgent



hyperparameters = \
    {
        'highway': {

            'natural_adversarial_model':
                {
                    "gail": {'model_name': 'NaturalAdversarialModel', 'max_action': [np.pi, 10], 'state_dim': 13, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 1, 'num_threads': 6, 'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "NGSIM_env/data/trajectory_set/highway", "supervise_model_name": 'Gail_1', "supervise_model_id": 0, "supervise_model_state_dim": 54, "supervise_model_action_dim": 2, 'supervise_use_running_state': False},
                    "diffusion": {'model_name': 'NaturalAdversarialModel', 'load_model_id': 359, 'max_action': [np.pi/4, 5], 'state_dim': 13, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 1000, 'num_threads': 6, 'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "NGSIM_env/data/trajectory_set/highway", "supervise_model_name": 'DiffusionAgent_1', "supervise_model_id": 3700, "supervise_model_state_dim": 54, "supervise_model_action_dim": 2, 'supervise_use_running_state': False, "supervise_model_T": 16},
                }
        },

        'intersection': {

            'natural_adversarial_model':
                {
                    "gail": {'model_name': 'NaturalAdversarialModel', 'max_action': [np.pi/2, 10],'load_model_id': 0, 'state_dim': 9, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 500, 'num_threads': 1 ,'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "NGSIM_env/data/trajectory_set/intersection","supervise_model_name": 'Gail_1', "supervise_model_id": 0, "supervise_model_state_dim": 56, "supervise_model_action_dim": 2, 'supervise_use_running_state': False, "supervise_model_T": 16},
                    "diffusion": {'model_name': 'NaturalAdversarialModel', 'load_model_id': 0,'max_action': [np.pi/2, 10], 'state_dim': 9, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 500, 'num_threads': 1 ,'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "NGSIM_env/data/trajectory_set/intersection","supervise_model_name": 'DiffusionAgent_5000', "supervise_model_id": 4899, "supervise_model_state_dim": 56, "supervise_model_action_dim": 2, 'supervise_use_running_state': False, "supervise_model_T": 16},
                }
        }
    }


def main(args):
    scenario_name = args.scenario
    policy = args.policy
    model = args.supervise_model if policy != "human_driving_policy" else args.model
    param = hyperparameters[scenario_name][policy][model]
    if args.num_threads is not None:
        param['num_threads'] = args.num_threads
    param['load_model_id'] = args.load_model_id

    if policy == 'natural_adversarial_model':

        agent = NaturalAdversarialAgent(args, param['state_dim'], param['action_dim'], param['max_action'], load_model_id=param['load_model_id'], num_threads=param['num_threads'],
                     max_iter_num=param['max_iter_num'], use_running_state=param['use_running_state'], lr=param['lr'],
                     save_dir=param['save_dir'], project_dir=project_dir, scenario=scenario_name, vehicle_dir=param['vehicle_dir'],
                     supervise_model_name=param['supervise_model_name'], supervise_model_id=param['supervise_model_id'], supervise_model_action_dim=param['supervise_model_action_dim'],
                     supervise_model_state_dim=param['supervise_model_state_dim'], supervise_use_running_state= param['supervise_use_running_state'], supervise_model_T=param['supervise_model_T'],train=False)
        agent.exp_with_policy(epochs=args.epochs, show=args.show, save_videos=args.save_video)


def make_args():
    args = ArgumentParser()
    args.add_argument('--policy', default='', type=str, required=True, help=""" It's one of three: human_driving_policy, adversarial_model and natural_adversarial_model.""")
    args.add_argument('--scenario', default='', type=str, required=True, help=""" It's one of two: highway and intersection.""")
    args.add_argument('--model', default='', type=str, required=True, help=""" It's one of three:ppo, trpo, a2c.""")
    args.add_argument('--supervise-model', default='', type=str, required=True, help="""It's one of two: gail, diffusion""")
    args.add_argument('--exp-name', default='exp_1', type=str)
    args.add_argument('--num-threads', default=None, type=int)
    args.add_argument('--epochs', default=1000, type=int)
    args.add_argument('--num-epochs', default=1000, type=int)
    args.add_argument('--seed', default=0, type=int)
    args.add_argument('--load-model-id', required=True,default=None, type=int)
    args.add_argument('--show', action='store_true', help="render")
    args.add_argument('--save-video', action='store_true', help="save videos")
    return args


if __name__ == '__main__':
    args = make_args().parse_args()
    main(args)