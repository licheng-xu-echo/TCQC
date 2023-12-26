from nff.data.dataset import QMMultiDataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir',type=str,default='/contest/B/dataset')
args = parser.parse_args()

root_dir = args.root_dir
#root = '/contest/B/dataset'
name = 'QMB_round2_train_proc_energy_drp_high.npy'
QMMultiDataset(root_dir, name, transform=None, pre_transform=None,train=True)