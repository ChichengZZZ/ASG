import os
import sys
current_dir = sys.path[0].replace("\\", "/")
project_dir = os.sep.join(current_dir.split('/')[:-2]).replace("\\", "/")
sys.path.append(project_dir)
from argparse import ArgumentParser

import numpy as np
from core.natural_adversarial_agent import NaturalAdversarialAgent

hyperparameters = \
    {
        'highway': {
                'gail': {'model_name': 'natural_adversarial_model', 'max_action': [np.pi / 4, 5], 'state_dim': 13, 'action_dim': 2, 'lr': 2e-4, 'batch_size': 2048, 'max_iter_num': 1000, 'num_threads': 4, 'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "data/ngsim/trajectory_set", "supervise_model_name": 'Gail/{}/exp_1', "supervise_model_id": 349, 'supervise_use_running_state': False ,"supervise_model_state_dim": 56, "supervise_model_action_dim": 2, "supervise_model_T": 16},
                'diffusion': {'model_name': 'natural_adversarial_model', 'max_action': [np.pi / 4, 5], 'state_dim': 13, 'action_dim': 2, 'lr': 3e-4,'batch_size': 2048, 'max_iter_num': 1000, 'num_threads': 6, 'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "data/ngsim/trajectory_set", "supervise_model_name": 'DiffusionAgent/exp_1', "supervise_model_id": 3700,'supervise_use_running_state': False, "supervise_model_state_dim": 56, "supervise_model_action_dim": 2, "supervise_model_T": 16},
                'ppo': {'model_name': 'natural_adversarial_model', 'max_action': [np.pi / 4, 5], 'state_dim': 13, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 1000, 'num_threads': 6, 'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "data/ngsim/trajectory_set", "supervise_model_name": 'DiffusionAgent/exp_1', "supervise_model_id": 310, 'supervise_use_running_state': False, "supervise_model_state_dim": 56, "supervise_model_action_dim": 2, "supervise_model_T": 16},
                'dqn': {'model_name': 'natural_adversarial_model', 'max_action': [np.pi / 4, 5], 'state_dim': 13, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 1000, 'num_threads': 6, 'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "data/ngsim/trajectory_set", "supervise_model_name": 'DiffusionAgent/exp_1', "supervise_model_id": 310, 'supervise_use_running_state': False, "supervise_model_state_dim": 56, "supervise_model_action_dim": 2, "supervise_model_T": 16},
        },

        'intersection': {
                'gail': {'model_name': 'natural_adversarial_model', 'max_action': [np.pi/2, 10], 'batch_size': 2048, 'state_dim': 13, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 1000, 'num_threads': 6 ,'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "data/interaction/trajectory_set", "supervise_model_name": 'Gail/{}/exp_1', "supervise_model_id": 400,'supervise_use_running_state': False, "supervise_model_state_dim": 56, "supervise_model_action_dim": 2, "supervise_model_T": 16},
                'diffusion': {'model_name': 'natural_adversarial_model', 'max_action': [np.pi/2, 10], 'batch_size': 2048,'state_dim': 13, 'action_dim': 2, 'lr': 3e-4, 'max_iter_num': 1000, 'num_threads': 6 ,'use_running_state': False, 'save_dir': 'examples/model_data', 'vehicle_dir': "data/interaction/trajectory_set", "supervise_model_name": 'DiffusionAgent/exp_1', "supervise_model_id": 4899, 'supervise_use_running_state': False,"supervise_model_state_dim": 56, "supervise_model_action_dim": 2, "supervise_model_T": 16},
        },
    }


def main(args):
    scenario_name = args.scenario
    model = args.supervise_model

    param = hyperparameters[scenario_name][model]
    if args.num_threads is not None:
        param['num_threads'] = args.num_threads

    param['max_iter_num'] = args.num_epochs

    param['supervise_model_id'] = args.supervise_model_id

    agent = NaturalAdversarialAgent(args, param['state_dim'], param['action_dim'], param['max_action'], batch_size=param['batch_size'], num_threads=param['num_threads'],
                 max_iter_num=param['max_iter_num'], use_running_state=param['use_running_state'], lr=param['lr'],
                 save_dir=param['save_dir'], project_dir=project_dir, scenario=scenario_name, vehicle_dir=param['vehicle_dir'],
                 supervise_model_name=param['supervise_model_name'], supervise_model_id=param['supervise_model_id'],
                 supervise_model_action_dim=param['supervise_model_action_dim'], supervise_model_state_dim=param['supervise_model_state_dim'],
                 supervise_model_T=param['supervise_model_T'], supervise_use_running_state=param['supervise_use_running_state'])
    agent.learn(debug=False)


def make_args():
    args = ArgumentParser()
    args.add_argument('--scenario', default='highway', type=str, help=""" It's one of two: highway and intersection.""")
    args.add_argument('--model', default='', type=str,  help=""" It's one of three:ppo, trpo, a2c.""")
    # args.add_argument('--update-mode', default='ppo', type=str, help=""" for ppo agent, ppo, ppo_v2.""")
    args.add_argument('--action-type', default='discrete', type=str,  help=""" continue or discrete""")
    args.add_argument('--supervise-model', default='', type=str, help="""It's one of two: gail, diffusion""")
    args.add_argument('--supervise-model-update-mode', default='', type=str, help="""for gail, ppo, trpo, a2c""")
    args.add_argument('--supervise-model-id', default=None, type=int, help="""model epoch number.""")
    args.add_argument('--num-epochs', default=None, type=int, help="""train epoch number.""")
    args.add_argument('--exp-name', default='exp_1', type=str)
    args.add_argument('--policy-type', default='NAM', type=str, help='for distinguishing between nat adv model and adv model')
    args.add_argument('--num-threads', default=2, type=int)
    args.add_argument('--seed', default=0, type=int, help="随机种子")
    args.add_argument('--resume', action='store_true', default=False, help=""" training with the most recently save model. """)
    args.add_argument('--debug', action='store_true', default=False, help=""" debug mode. """)
    args.add_argument('--upload', action='store_true', help="upload exp data to wandb server.")
    # for ddpg
    args.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    args.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    args.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')

    # 重置机制参数
    args.add_argument('--reset-interval', default=500, type=int, help="步数间隔，达到后重置网络层")
    args.add_argument('--reset-network', default=3, type=int, help="需要重置的网络层数")
    args.add_argument('--replay-buffer-capacity', default=10000, type=int, help="Replay Buffer的容量")
    return args


if __name__ == '__main__':
    args = make_args().parse_args()
    main(args)