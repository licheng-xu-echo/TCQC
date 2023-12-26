import torch,glob
from tqdm import tqdm
from ..data.features import ATOM_ENERGY,HA2KCAL
from nff.nn.models import DimeNetPP
from dig.threedgraph.method import SphereNet
from torch_geometric.data import DataLoader
def E_model_eval(E_model,model_path,dataload,device='cpu',energy_and_force=True):

    checkpoint = torch.load(model_path,map_location=device)
    E_model.load_state_dict(checkpoint['model_state_dict'])
    E_model.to(device=device)

    E_model.eval()
    preds_force = torch.Tensor([])
    preds = torch.Tensor([])
    for step, batch_data in enumerate(tqdm(dataload)):
        batch_data = batch_data.to(device)
        
        out_res = E_model(batch_data)
        out = out_res['energy'].detach_().cpu().double()
        if energy_and_force:
            #force = -out_res['energy_grad'].detach_().cpu()
            force_ = torch.zeros(batch_data.pos.shape)
            #force_ = torch.ones(batch_data.pos.shape) * 1000
            #print(force.shape)
            #print(force_.shape)
            preds_force = torch.cat([preds_force,force_], dim=0)
        base_e = []
        for m_idx in range(len(batch_data.ptr)-1):
            start_idx = batch_data.ptr[m_idx]
            end_idx = batch_data.ptr[m_idx+1]
            z_m = batch_data.z[start_idx:end_idx]
            base_e.append(sum([ATOM_ENERGY['B3LYP/DEF2TZVP'][int(item)] for item in z_m])*HA2KCAL)
        base_e = torch.tensor(base_e,dtype=torch.float64).view(-1,1)
        #print(base_e)
        #print(out)
        out += base_e
        preds = torch.cat([preds, out], dim=0)

    return preds,preds_force
def E_model_eval_BG(E_model,model_path_0,model_path_1,dataload,device='cpu',energy_and_force=True):

    checkpoint_0 = torch.load(model_path_0,map_location=device)
    E_model.load_state_dict(checkpoint_0['model_state_dict'])
    E_model.to(device=device)

    E_model.eval()
    preds_force = torch.Tensor([])
    preds_0 = torch.Tensor([])
    for step, batch_data in enumerate(tqdm(dataload)):
        batch_data = batch_data.to(device)
        
        out_res = E_model(batch_data)
        out = out_res['energy'].detach_().cpu().double()
        if energy_and_force:
            force_ = torch.zeros(batch_data.pos.shape)
            preds_force = torch.cat([preds_force,force_], dim=0)
        base_e = []
        for m_idx in range(len(batch_data.ptr)-1):
            start_idx = batch_data.ptr[m_idx]
            end_idx = batch_data.ptr[m_idx+1]
            z_m = batch_data.z[start_idx:end_idx]
            base_e.append(sum([ATOM_ENERGY['B3LYP/DEF2TZVP'][int(item)] for item in z_m])*HA2KCAL)
        base_e = torch.tensor(base_e,dtype=torch.float64).view(-1,1)
        #print(base_e)
        #print(out)
        out += base_e
        preds_0 = torch.cat([preds_0, out], dim=0)

    checkpoint_1 = torch.load(model_path_1,map_location=device)
    E_model.load_state_dict(checkpoint_1['model_state_dict'])
    E_model.to(device=device)

    E_model.eval()
    preds_1 = torch.Tensor([])
    for step, batch_data in enumerate(tqdm(dataload)):
        batch_data = batch_data.to(device)
        
        out_res = E_model(batch_data)
        out = out_res['energy'].detach_().cpu().double()
        if energy_and_force:
            force_ = torch.zeros(batch_data.pos.shape)
            preds_force = torch.cat([preds_force,force_], dim=0)
        base_e = []
        for m_idx in range(len(batch_data.ptr)-1):
            start_idx = batch_data.ptr[m_idx]
            end_idx = batch_data.ptr[m_idx+1]
            z_m = batch_data.z[start_idx:end_idx]
            base_e.append(sum([ATOM_ENERGY['B3LYP/DEF2TZVP'][int(item)] for item in z_m])*HA2KCAL)
        base_e = torch.tensor(base_e,dtype=torch.float64).view(-1,1)
        out += base_e
        preds_1 = torch.cat([preds_1, out], dim=0)
    preds = preds_0*0.532+preds_1*0.468
    return preds,preds_force
def Multi_E_model_eval(model_dir,dataset,device='cpu'):
    model_chkpt_path_lst = sorted(glob.glob(f'{model_dir}/valid_checkpoint-*.pt'))
    all_preds = []
    for idx,model_chkpt_path in enumerate(model_chkpt_path_lst):
        print(f'Model {idx} checkpoint {model_chkpt_path}')
        if 'dimenetpp' in model_chkpt_path:
            model = DimeNetPP(energy_and_force=False,ret_res_dict=False)
            E_model_batch_size = 168
        elif 'spherenet' in model_chkpt_path:
            model = SphereNet(energy_and_force=False)
            E_model_batch_size = 80
        if 'tzvp' in model_chkpt_path:
            method = 'B3LYP/DEF2TZVP'
            unit_conv = 627.5
        elif 'svp' in model_chkpt_path:
            method = 'B3LYP/DEF2SVP'
            unit_conv = 627.509
        dataload = DataLoader(dataset, E_model_batch_size, shuffle=False)
        checkpoint = torch.load(model_chkpt_path,map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device=device)
        model.eval()
        if idx == 0:
            preds_force = torch.Tensor([])
        preds_energy = torch.Tensor([])
        for step, batch_data in enumerate(tqdm(dataload)):
            batch_data = batch_data.to(device)
            if idx == 0:
                force_ = torch.zeros(batch_data.pos.shape)
                preds_force = torch.cat([preds_force,force_], dim=0)
            out_res = model(batch_data)
            out = out_res.detach_().cpu().double()
            pred_base_e = []
            for m_idx in range(len(batch_data.ptr)-1):
                start_idx = batch_data.ptr[m_idx]
                end_idx = batch_data.ptr[m_idx+1]
                z_m = batch_data.z[start_idx:end_idx]
                pred_base_e.append(sum([ATOM_ENERGY[method][int(item)] for item in z_m])*unit_conv)
            pred_base_e = torch.tensor(pred_base_e,dtype=torch.float64).view(-1,1)
            out += pred_base_e
            preds_energy = torch.cat([preds_energy, out], dim=0)
        all_preds.append(preds_energy)
    return all_preds[0] * 0.330 + all_preds[1] * 0.265 + all_preds[2] * 0.141 + all_preds[3] * 0.130 + all_preds[4] * 0.134,preds_force
def E_model_eval_ans_gen(E_model,model_path,dataload,device='cpu',
                         method='B3LYP/DEF2TZVP',unit_conv=627.5):
    checkpoint = torch.load(model_path,map_location=device)
    E_model.load_state_dict(checkpoint['model_state_dict'])
    E_model.to(device=device)
    E_model.eval()
    preds = torch.Tensor([])
    truths = torch.Tensor([])
    for step, batch_data in enumerate(tqdm(dataload)):
        batch_data = batch_data.to(device)
        truth = batch_data.y.cpu()
        out_res = E_model(batch_data)
        #print(out_res)
        out = out_res.detach_().cpu().double()

        truth_base_e = []
        pred_base_e = []
        for m_idx in range(len(batch_data.ptr)-1):
            start_idx = batch_data.ptr[m_idx]
            end_idx = batch_data.ptr[m_idx+1]
            z_m = batch_data.z[start_idx:end_idx]
            truth_base_e.append(sum([ATOM_ENERGY['B3LYP/DEF2TZVP'][int(item)] for item in z_m])*627.5)
            pred_base_e.append(sum([ATOM_ENERGY[method][int(item)] for item in z_m])*unit_conv)
        truth_base_e = torch.tensor(truth_base_e,dtype=torch.float64).view(-1,1)
        pred_base_e = torch.tensor(pred_base_e,dtype=torch.float64).view(-1,1)
        truth = torch.tensor(truth,dtype=torch.float64).view(-1,1)
        out += pred_base_e
        truth += truth_base_e
        preds = torch.cat([preds, out], dim=0)
        truths = torch.cat([truths, truth],dim=0)
    truths = truths.view(-1).numpy()
    preds = preds.view(-1).numpy()
    return truths,preds