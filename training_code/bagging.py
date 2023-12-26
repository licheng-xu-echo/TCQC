import numpy as np
import argparse,glob,os
from nff.nn.models import DimeNetPP
from dig.threedgraph.method import SphereNet
from nff.data.dataset import QMDataset
from torch_geometric.data import DataLoader
from nff.eval.eval_model import E_model_eval_ans_gen

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir',type=str,default='/contest/xlc/')
parser.add_argument('--data_split_seed',type=int,default=42)
parser.add_argument('--train_set_size',type=int,default=8000000)
parser.add_argument('--valid_set_size',type=int,default=1000000)
parser.add_argument('--E_model_dir',type=str,default='../final/energy_model_3')
parser.add_argument('--E_model_batch_size',type=int,default=32)
parser.add_argument('--device',type=str,default='cuda:0')
parser.add_argument('--dataset_type',type=str,default='valid')
parser.add_argument('--stage',type=int,default=1)   ## 1 模型推理，2 Bagging
parser.add_argument('--result_save_dir',type=str,default='./bagging')
args = parser.parse_args()
def main():
    root_dir = args.root_dir
    data_split_seed = args.data_split_seed
    train_set_size = args.train_set_size
    valid_set_size = args.valid_set_size
    E_model_dir = args.E_model_dir
    E_model_batch_size = args.E_model_batch_size
    device = args.device
    dataset_type = args.dataset_type
    stage = args.stage
    result_save_dir = args.result_save_dir
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)
    if stage == 1:
        
        dataset = QMDataset(root=root_dir,name='QMB_round2_train_proc_energy_drp_high.npy')
        split_idx = dataset.get_idx_split(len(dataset), train_size=train_set_size, valid_size=valid_set_size, seed=data_split_seed)
        print(f'Train: {len(split_idx["train"])}, Valid: {len(split_idx["valid"])}, test: {len(split_idx["test"])}')
        train_dataset, valid_dataset,test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']],dataset[split_idx['test']]
        valid_loader = DataLoader(valid_dataset, E_model_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, E_model_batch_size, shuffle=False)
        E_model_path_lst = sorted(glob.glob(f'{E_model_dir}/valid_*.pt'))
        for i,E_model_path in enumerate(E_model_path_lst):
            post = E_model_path.split('.')[-2].split('-')[-1]
            if os.path.exists(f'{result_save_dir}/{dataset_type}-{post}.npy'):
                continue
            if 'dimenetpp' in E_model_path:
                model = DimeNetPP(energy_and_force=False,ret_res_dict=False)
            elif 'spherenet' in E_model_path:
                model = SphereNet(energy_and_force=False)
            print(f'[{i+1} / {len(E_model_path_lst)}] Energy model evaluation on {dataset_type} dataset...')
            
            if dataset_type == 'valid':
                if 'tzvp' in E_model_path:
                    E_truth,E_pred = E_model_eval_ans_gen(E_model=model,model_path=E_model_path,dataload=valid_loader,device=device)
                elif 'svp' in E_model_path:
                    E_truth,E_pred = E_model_eval_ans_gen(E_model=model,model_path=E_model_path,dataload=valid_loader,device=device,method='B3LYP/DEF2SVP',unit_conv=627.509)
            elif dataset_type == 'test':
                if 'tzvp' in E_model_path:
                    E_truth,E_pred = E_model_eval_ans_gen(E_model=model,model_path=E_model_path,dataload=test_loader,device=device)
                elif 'svp' in E_model_path:
                    E_truth,E_pred = E_model_eval_ans_gen(E_model=model,model_path=E_model_path,dataload=test_loader,device=device,method='B3LYP/DEF2SVP',unit_conv=627.509)
            E_mae = np.abs(E_truth-E_pred).mean()
            print(f'[{i+1} / {len(E_model_path_lst)}] MAE of energy is: {E_mae:.4f}')
            np.save(f'{result_save_dir}/{dataset_type}-{post}.npy',E_pred)
            if not os.path.exists(f'{result_save_dir}/{dataset_type}-truth.npy'):
                np.save(f'{result_save_dir}/{dataset_type}-truth.npy',E_truth)
    elif stage == 2:
        truth_arr = np.load(f'{result_save_dir}/{dataset_type}-truth.npy')
        pred_lst = []
        pred_files = sorted(glob.glob(f'{result_save_dir}/{dataset_type}*.npy'))
        for pred_file in pred_files:
            if not 'dimenetpp' in pred_file and not 'spherenet' in pred_file:
                continue
            #if 'spherenet' in pred_file:
            #    continue
            print(f'Model checkpoint: {os.path.basename(pred_file)}')
            pred_lst.append(np.load(pred_file))
        
        weight_range = 100
        best_mae = float('inf')
        best_weight = (20,0,80,0,0)
        # best 33 : 27 : 14 : 13 : 13
        # best 330 : 265 : 141 : 130 : 134
        for i in range(320,340):
            for j in range(260,280):
                for k in range(130,150):
                    for l in range(120,140):
                        m = weight_range - i -j - k -l
                    #l = weight_range - i - j - k
                        if m < 0:
                            continue
                        pred_arr = (pred_lst[0] * i + pred_lst[1] * j + pred_lst[2] * k + pred_lst[3] * l + pred_lst[4] * m) / weight_range
                        mae = np.abs(pred_arr - truth_arr).mean()
                        if mae < best_mae:
                            print(f'Weight {i} : {j} : {k} : {l} : {m} MAE is {mae:.4f}')
                            best_mae = mae
                            best_weight = (i,j,k,l,m)
        print(f'Best weight is {best_weight[0]} : {best_weight[1]} : {best_weight[2]} : {best_weight[3]} : {best_weight[4]}, MAE is: {best_mae:.4f}')
if __name__ == '__main__':
    main()