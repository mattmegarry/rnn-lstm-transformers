#!/bin/bash

# Download and install miniconda
wget -P ~/ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/Miniconda3-latest-Linux-x86_64.sh
~/Miniconda3-latest-Linux-x86_64.sh -b

# Configure miniconda
export PATH=~/miniconda3/bin:$PATH 
conda init

# Disable auto activate base env 
conda config --set auto_activate_base false

# Create conda environment
conda create --name gigglers-env python=3.8 -y

# Activate environment
conda activate gigglers-env

# Install packages
conda install pipx
pipx run nvitop