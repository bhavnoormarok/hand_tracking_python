conda create --name env_hand python=3.8
conda activate env_hand


# install libraries only available with conda before installing any packages with pip (https://www.anaconda.com/blog/using-pip-in-a-conda-environment)
conda install -c conda-forge igl

# upgrade pip
pip install -U pip

# jax (For CUDA 11.1 or newer, use cuda111. The same wheel should work for CUDA 11.x releases from 11.1 onwards)
# Installs the wheel compatible with Cuda 11 and cudnn 8.2 or newer.
# pip install 'jax[cuda11_cudnn82]' -f https://storage.googleapis.com/jax-releases/jax_releases.html
# quotes ('') around jax[cuda11_cudnn82] are required on zsh
# on cpu use
pip install --upgrade "jax[cpu]"

# numerical libraries
pip install scipy scikit-learn

# hyperparameter tuning
pip install optuna

# image, video, graphics io
pip install scikit-image opencv-python open3d
pip install kaleido plotly
pip install ipympl
pip install mediapy
pip install colorcet
pip install moderngl moderngl-window


# utility
pip install tqdm
pip install numba

# chumpy required for loading MANO model
pip install chumpy

# hyperparameter optimization
pip install optuna