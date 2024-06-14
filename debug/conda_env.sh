clear;
# python 3.11
conda install conda-forge::kornia --yes
conda install conda-forge::dominate --yes
conda install conda-forge::tensorboard --yes
conda install anaconda::scipy --yes
conda install conda-forge::trimesh --yes
cd nvdiffrast/
pip install .
cd ..
pip install scikit-image
conda install -c menpo opencv --yes
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
