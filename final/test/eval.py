#%%
import argparse,glob,os
import numpy as np
from nff.data.dataset import QMDataset
from nff.eval.eval_model import Multi_E_model_eval
def gen_ans(preds,preds_force,test_dataset,init_idx=950000,output_dir='./submit'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ans = ['id,energy,force']

    cumm_at_num = 0
    for idx,energy in enumerate(preds):
        data = test_dataset[idx]
        energy = float(energy[0])
        at_num = len(data['z'])
        force = preds_force[cumm_at_num:cumm_at_num+at_num]
        cumm_at_num += at_num
        force_str = ','.join([str(float(f)) for f in force.view(-1)])
        
        ans.append(f'{init_idx+idx},{energy},\"{force_str}\"')
    with open(f'{output_dir}/submission.csv','w') as fw:
        fw.writelines('\n'.join(ans))
    print(f'[INFO] Prediction result is saved in {output_dir}/submission.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--tcdata_path',type=str,default='./tcdata')
parser.add_argument('--E_model_batch_size',type=int,default=16)
parser.add_argument('--use_F_model',type=bool,default=False)
parser.add_argument('--device',type=str,default='cuda:0')
parser.add_argument('--output_dir',type=str,default='./app')
parser.add_argument('--E_model_dir',type=str,default='../energy_model_3')
parser.add_argument('--BG',type=bool,default=True)
args = parser.parse_args()

def main():
    tcdata_path = args.tcdata_path
    E_model_batch_size = args.E_model_batch_size
    use_F_model = args.use_F_model
    device = args.device
    output_dir = args.output_dir
    E_model_dir = args.E_model_dir
    BG = args.BG
    test_data_files = sorted(glob.glob(f'{tcdata_path}/QMB_round2_test*.npy'),key=lambda x:int(x.split('.')[-2].split('_')[-1]))
    all_data = []
    for npy_file in test_data_files:
        data = np.load(npy_file,allow_pickle=True)
        all_data.append(data)
    all_data = np.concatenate(all_data)
    test_qmdataset = QMDataset(root=tcdata_path,name='QMB_round2_test_230725.npy',raw_data=all_data,train=False)
    preds,preds_force = Multi_E_model_eval(E_model_dir,test_qmdataset,device=device)
    gen_ans(preds,preds_force,test_qmdataset,init_idx=all_data[0]['mol_name'],output_dir=output_dir)
if __name__ == '__main__':
    main()