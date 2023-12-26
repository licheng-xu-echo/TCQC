# TCQC

**T**ian**C**hi **Q**uantum **C**hemistry (TCQC). This is the code repository for our participation in the [&#34;First World Scientific Intelligence Competition: Quantum Chemistry Track.&#34;](https://tianchi.aliyun.com/competition/entrance/532115) In this competition, after going through the preliminary round, semifinal, and final, our team "Qizhen Chemistry" (**启真化学**) achieved third place.

# Algorithm

Our project is almost based on [DIG](https://github.com/divelab/DIG) and [NeuralForceField](https://github.com/learningmatter-mit/NeuralForceField).

We utilized [DimeNet++](https://arxiv.org/abs/2011.14115) for energy fitting. The predicted forces were obtained by taking the negative derivative of the predicted energy with respect to the coordinates. Taking training efficiency into account, we did not incorporate the predictive error of forces into the model training during the preliminary stage. In the later stages of the contest, both force and energy prediction errors will be integrated into the training process.

Due to the complex molecular compositions in the dataset provided by this contest, the range of molecular energies is around ~ $10^6$ kcal/mol. We believe this energy range is unfavorable for energy fitting. To mitigate the energy disparities caused by varying elemental compositions within molecules, we subtracted the atomic energies from the molecular energies. In the current stage, both the training and test sets consist only of eight elements: H, C, N, O, F, P, S, and Cl. Calculating the energy of these individual atoms at B3LYP/def2tzvp (can be any other method and basis set) level of theory takes **just a few seconds**, **which can be considered negligible**.

After subtracting the atomic energies from the molecular energies in the training set, our training objective effectively became the molecular formation energy. Upon predicting the energy on the test set, adding back the atomic energies yields the energy of the test set molecules. Through this approach, we successfully narrowed down our target range from $10^6$ to $10^3$, significantly enhancing the convergence efficiency of the model.

# Conda environment

We run our code in the nvidia container and the nvidia **CUDA version is 11.3**. We highly recommend creating a conda environment to run the code. To do that, clone this project, cd into this project folder and use the following command to create the `tcqc` conda environment:

```
conda env create -f ./code/environment.yaml
```

If creating a conda environment from the "environment.yaml" file doesn't work, please manually create a python 3.8 conda environment named as "tcqc" and install the packages listed below.

```
conda create -n tcqc python=3.8
conda activate tcqc
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install ase networkx pymatgen sympy rdkit e3fp tensorboard dive-into-graphs
```

# Installation

Execute the command within the project directory ('TCQC' folder), to install NeuralForceField as a package:

```
pip install .
```

# Usage

To generate the pre-processed PyTorch data file, execute the code within the 'training_code' folder:

```
python data.py --root_dir <path where training data is saved> 
```

To train the molecular energy prediction model, execute the following code:

```
python energy_model.py --root_dir <path where training data is saved> --E_batch_size 256 --lr_decay_step 10 --train_energy_epoch 300
```

To enhance model performance, we trained three energy models, and subsequently applied a bagging approach to aggregate their predictive outputs.

```
python bagging.py --root_dir <path where training data is saved> --E_model_dir ./energy_model --stage 1 
python bagging.py --root_dir <path where training data is saved> --E_model_dir ./energy_model --stage 2 
```

