#conda remove -n torch --all
conda create -n torch python=3.7 anaconda -y
conda activate torch
# for cpu: conda install pytorch==1.9.0 torchvision -c pytorch -y
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c conda-forge -y
# conda install -c conda-forge visdom -y
# python datasets.py cifar10 "none" -o "none"
python train.py DCGAN -c config_dog120.json --restart -n dogs120 --np_vis
# python train.py PGAN -c config_cifar10.json -n dogs120 --np_vis