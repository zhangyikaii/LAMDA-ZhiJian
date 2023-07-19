import os
import random

from zhijian.models.utils import gpu_state

import time
from datetime import datetime
import json

gpus = '0,1,2,3'

space_hold = 12000
space_for_shixiong = 0
polling_interval = {'success': 20, 'fail': 20}
manual_exec_dicts = [
    ]

base_dir = '/data/zhangyk/models/best_kel_logs/new'

yaml_list = ['vpt_deep', 'vpt_shallow', 'adapter', 'ssf', 'lora', 'fact_tt', 'fact_tk', 'convpass', 'linear_eval', 'partial_1', 'partial_2', 'partial_4', 'finetune']

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

updated_general_params = ['lr', 'wd', 'optimizer', 'mom', 'lr-scheduler']
updated_params = {
    'adapter': {
        'adapter-dropout': 'dropout',
        'adapter-reduction-dim': 'reduction_dims'
    },
    'ssf': {},
    'lora': {
        'lora-dropout': 'dropout',
        'lora-dim': 'lora_dims'
    },
    'fact_tt': {
        'fact-tt-scale': 'scale',
        'fact-tt-dim': 'Fact_tt_dims'
    },
    'fact_tk': {
        'fact-tk-scale': 'scale',
        'fact-tk-dim': 'Fact_tk_dims'
    },
    'convpass': {
        'convpass-dim': 'dim',
        'convpass-scale': 'scale',
        'convpass-drop-path': 'drop_path',
        'convpass-xavier-init': 'xavier_init'
    },
    'vpt_deep': {
        'vpt-initiation': 'initiation',
        'vpt-project': 'project',
        'vpt-deep': 'deep',
        'vpt-dropout': 'dropout',
        'vpt-num-tokens': 'num_tokens'
    },
    'vpt_shallow': {
        'vpt-initiation': 'initiation',
        'vpt-project': 'project',
        'vpt-deep': 'deep',
        'vpt-dropout': 'dropout',
        'vpt-num-tokens': 'num_tokens'
    }
}


command_list = []

for cur_method in yaml_list:
    for cur_dataset in data[cur_method].keys():
        cur_time_str = data[cur_method][cur_dataset]
        if cur_time_str == '-1':
            continue
        cur_json = os.path.join(base_dir, cur_time_str, 'configs.json')

        if not os.path.isfile(cur_json):
            print('NOT FOUND', cur_method, cur_dataset, cur_json)
            continue

        loaded_data = load_json(cur_json)

        command = f'python main.py --model timm.vit_base_patch16_224_in21k --config models/configs/{cur_method}.yaml --dataset {cur_dataset} --dataset-dir /data/zhangyk/data/petl --verbose --test-all --max-epoch 100 --initial-checkpoint /data/zhangyk/data/petl/model/ViT-B_16.npz '
        for i_param in updated_general_params:
            command += f'--{i_param} {loaded_data[i_param.replace("-", "_")]} '
        for i_param in updated_params[cur_method].keys():
            command += f'--{i_param} {loaded_data[updated_params[cur_method][i_param]]} '
        command_list.append(command)



def exec_args(exec_dict, is_first, cur_command):
    gpu_available = ''
    while len(gpu_available) == 0:
        if not is_first:
            time.sleep(polling_interval['fail'])
        gpu_space_available = gpu_state(gpus, get_return=True)
        gpu_max = max(gpu_space_available.items(), key=lambda x:x[1])
        if gpu_max[1] - space_for_shixiong >= space_hold:
            gpu_available = gpu_max[0]

    exec_dict['gpu'] = gpu_available
    exec_dict['time-str'] = datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    command = f'nohup {cur_command} ' + ' '.join([f'--{param_name} {param_value}' for param_name, param_value in exec_dict.items() if param_name != 'random_grid_time'])
    command += f' >> ./nohup_logs/{exec_dict["time-str"]}.log 2>&1 &'

    """
    """

    log_str = f'{exec_dict["time-str"]} | {exec_dict} ({gpu_available}).'
    print(log_str)
    print(command, end='\n\n')
    os.system(command)

    time.sleep(polling_interval['success'] + random.randint(1, 20))


def main():
    is_first = True

    for cur_command in command_list:
        cur_exec_dict = {}
        exec_args(cur_exec_dict, is_first, cur_command)
        if is_first:
            is_first = False

if __name__ == '__main__':
    main()
