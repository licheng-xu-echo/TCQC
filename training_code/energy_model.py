import argparse,torch
from nff.nn.models import DimeNetPP
from nff.data.dataset import QMDataset,NewQMDataset
from nff.train.run import run
from nff.train.metrics import ThreeDEvaluator
#from nff.nn.models.torchmdnet.model import create_model
#from nff.nn.models.torchmdnet.params import TENSORNET_QM9_NO_F_PARM,ET_QM9_NO_F_PARM,TENSORNET_MD17_NO_F_PARM,ET_MD17_NO_F_PARM

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir',type=str,default='/tcdata')
parser.add_argument('--data_split_seed',type=int,default=42)
parser.add_argument('--train_set_size',type=int,default=8000000)
parser.add_argument('--valid_set_size',type=int,default=1000000)
parser.add_argument('--train_energy_epoch',type=int,default=200)
parser.add_argument('--E_model_save_dir',type=str,default='./energy_model')
parser.add_argument('--log_dir',type=str,default='')
parser.add_argument('--E_batch_size',type=int,default=256)
parser.add_argument('--lr_decay_step',type=int,default=15)
parser.add_argument('--model',type=str,default='dimenetpp')
parser.add_argument('--device',type=str,default='cuda:0')
parser.add_argument('--checkpoint',type=str,default='')

args = parser.parse_args()
device = args.device if torch.cuda.is_available() else 'cpu'

def main():
    root_dir = args.root_dir
    train_set_size = args.train_set_size
    valid_set_size = args.valid_set_size
    train_energy_epoch = args.train_energy_epoch
    data_split_seed = args.data_split_seed
    E_model_save_dir = args.E_model_save_dir
    E_batch_size = args.E_batch_size
    model_type = args.model.lower()
    checkpoint = args.checkpoint
    optimizer_state_dict = None
    scheduler_state_dict = None
    lr_decay_step = args.lr_decay_step
    if checkpoint != '':
        checkpoint = torch.load(checkpoint,map_location='cpu')
    log_dir = args.log_dir if args.log_dir != '' else E_model_save_dir
    dataset = QMDataset(root=root_dir,name='QMB_round2_train_proc_energy_drp_high.npy')
    split_idx = dataset.get_idx_split(len(dataset), train_size=train_set_size, valid_size=valid_set_size, seed=data_split_seed)
    print(f'Train: {len(split_idx["train"])}, Valid: {len(split_idx["valid"])}, test: {len(split_idx["test"])}')
    train_dataset, valid_dataset = dataset[split_idx['train']], dataset[split_idx['valid']]
    if len(split_idx['test']) == 0:
        test_dataset = valid_dataset
    else:
        test_dataset = dataset[split_idx['test']]
    if model_type == 'dimenetpp':
        #model = DimeNetPP(energy_and_force=False,ret_res_dict=False,num_layers=8,cutoff=5,int_emb_size=128)
        model = DimeNetPP(energy_and_force=False,ret_res_dict=False)
        if checkpoint != '':
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            scheduler_state_dict = checkpoint['scheduler_state_dict']
            scheduler_state_dict['step_size'] = lr_decay_step
    #elif model_type == 'torchmd_et_qm9':
    #    model = create_model(ET_QM9_NO_F_PARM)
    #elif model_type == 'torchmd_tensornet_qm9':
    #    model = create_model(TENSORNET_QM9_NO_F_PARM)
    #elif model_type == 'torchmd_et_md17':
    #    model = create_model(ET_MD17_NO_F_PARM)
    #elif model_type == 'torchmd_tensornet_md17':
    #    model = create_model(TENSORNET_MD17_NO_F_PARM)
    else:
        print(f'[ERROR] Unsupport model type {model_type}')
        return 
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()
    run3d = run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model,loss_func, evaluation,epochs=train_energy_epoch,
              batch_size=E_batch_size,vt_batch_size=E_batch_size,save_dir=E_model_save_dir,energy_and_force=False,log_dir=log_dir,
              lr=0.0005,lr_decay_factor=0.5,lr_decay_step_size=lr_decay_step,optimizer_state_dict=optimizer_state_dict,scheduler_state_dict=scheduler_state_dict) 
    ## default lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15

if __name__ == '__main__':
    main()