#!/bin/bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source $HOME/.poetry/env
sudo yum install git -y
git clone https://github.com/YannBerthelot/DeepRL.git
cd DeepRL
git checkout continuous_A2C
sudo amazon-linux-extras install python3.8 -y
sudo yum install python38 python38-devel python38-libs python38-tools -y
