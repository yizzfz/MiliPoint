  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
  conda install toml pyg scipy -c pyg -y
  conda install toml -y
  python -m pip install optuna pytorch-lightning
  python -m pip install opencv-python
  python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
