# MiliPoint (A Point Cloud Dataset for mmWave Radar)

![alt text](magnemite.png)

[Magnemite](https://bulbapedia.bulbagarden.net/wiki/Magnemite_(Pok%C3%A9mon)) (Japanese: コイル Coil) is a dual-type Electric/Steel Pokémon introduced in Generation I. Prior to Generation II, it was a pure Electric-type Pokémon. This Pokemon specialises in generating waves.

For more, you can watch this [video](https://www.youtube.com/watch?v=Vi-Lfm0-AXg&ab_channel=FalseSwipeGaming).

---

## Dataset Download and Preparation

Clone the project.

```
git clone https://github.com/yizzfz/MiliPoint.git
cd MiliPoint
```

Download `MiliPoint_data.zip` from [Google drive](https://drive.google.com/file/d/1rq8yyokrNhAGQryx7trpUqKenDnTI6Ky/view?usp=drive_link), unzip all contents to `data/raw/`:
```
MiliPoint
└─data
  └─raw
    ├─1.pkl
    ├─2.pkl
    ├─...
```

---

## Simple Usage: Using on only the Dataset

We provide a direct `pip` installation by running the following

`${CUDA}` should be replaced by either cpu, cu116, or cu117 depending on your PyTorch installation.

```bash
python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
python -m pip install .
```


Make a directory  `mkdir data/processed/mmr_kp`, the following code shows how to run a dataset using the pip-installed package.

```python
from mmrnet.dataset import get_dataset
from mmrnet.models import model_map

import torch

dataset_config = {
		'seed': 20,
		'train_split': 0.8,
		'val_split': 0.1,
		'test_split': 0.1,
		'stacks': 3,
		'zero_padding': 'per_data_point'
}
loaders = get_dataset(
		'mmr_kp',
		batch_size=32,
		workers=0,
		mmr_dataset_config=dataset_config)
train_loader, val_loader, test_loader, info = loaders
# each dataset element in the loader is a tuple of (data, label)

print(train_loader.dataset[0][0].shape)
# Input point cloud size: torch.Size([66, 3])

print(train_loader.dataset[0][1].shape)
# Output keypoint size torch.Size([9, 3])
```

---

## Using the Datasets, Models and our Training and Evaluation Frameworks

Installation by `./install_mmrnet.sh`, then you can follow the commands recorded below to use different functionalities.

---

## Commands
Format: `./mm <train/test> <task> <model> <-args>` 
> Replace `./mm` with `python mm` if using Windows.

### Train

* Example Training run
	```bash
	./mm train mmr_kp dgcnn
		# cli argument for the saving directory
		\ --save session_name 
		# A config file that defines how to construct a dataset
		\ -config ./configs/keypoints/mmr_keypoints_stack_3_point.toml
		# to use GPU
		\ -a gpu
	```

	The config file defines the dataset building, for example `./configs/keypoints/mmr_keypoints_stack_5_point.toml` is:

  ```toml
	seed = 20
	processed_data = 'data/processed/mmr_keypoints/stacks_3_padd_point_task_keypoints.pkl'
	train_split = 0.8
	val_split = 0.1
	test_split = 0.1
	stacks = 3
	zero_padding = 'per_data_point'
	```

	These parameters will change the parameters in the dataset class (e.g. `MMRKeypointData`).

* Command line arguments have a higher priority

	```bash
 	./mm train mmr mlp --save session_name -config ./configs/keypoints/mmr_keypoints_stack_3_stack.toml --dataset_stacks 5
	```

	- In this case, `stacks` would be set to be `5` instead of `3` (defined in `.toml`), because command line arguments have a higher priority.
	
### Test

```bash
./mm eval mmr_kp dgcnn
	\ --load ckpt_name 
	\ -config ./configs/keypoints/mmr_keypoints_stack_3_stack.toml
	# to visualize test result as a video, only applicable for keypoint
	\ -v 
```

--- 

## Dataset hyperparameters

- `--dataset_stacks` (in `cli`) or `stacks` (in `toml`): this defines how many previous frames we stack together with the current frame.

- `--dataset_zero_padding` (`cli`) or `zero_padding` (`toml`): there are two styles of padding: 1) pad per frame (`--dataset_zero_padding data_point`) or 2) pad per stack (`--dataset_zero_padding stack`)

- `--dataset_num_keypoints` (`cli`) or `num_keypoints` (`toml`): the number of keypoints used to represent the human skeleton. `low` uses 9 points (left and right shoulders, elbows, waists, knees, and a head), `high` uses 18 points (the [COCO18 format](https://www.stereolabs.com/docs/body-tracking/)).

* `processed_data`: this declares the data path for storing the processed data file, if the file exists, the processing step will be skipped. However, you can use `-d_forced_rewrite` or `--dataset_forced_rewrite` to force to rewrite the cached file.

- The rest hyperparameters are fairly straightforward and are in `mmr_keypoints.toml`

---

## Models
* MLP: simple network with dense layers only.
* DGCNN
* PointNet++ 
* PointTransformer
* PointMLP

The performance of these models are included in the paper.

---

## Tasks

### mmr_kp - Keypoint estimation 

* X: Point cloud (n, 3), Y: The 18 keypoints of the subject. 

* Train command example:
	```bash
	python mm train mmr_kp dgcnn --save mmr_dgcnn -config ./configs/keypoints/mmr_keypoints_stack_5_point.toml -w 0 -a gpu -m 300 -n -1
	```

* Test command example:
	```bash
	python mm test mmr_kp dgcnn --load mmr_dgcnn -config ./configs/keypoints/mmr_keypoints_stack_5_point.toml -w 0 -a gpu
	```


### mmr_iden - Identification

* X: Point cloud (n, 3), Y: The subject id (1 - 18).

* Train command example: 
	```bash
	python mm train mmr_iden dgcnn --save mmr_iden_dgcnn -config ./configs/iden/mmr_iden_stack_5_point.toml -w 0 -a gpu -m 300 -n -1
	```

* Test command example:
	```bash
	python mm test mmr_iden dgcnn --load mmr_iden_dgcnn -config ./configs/iden/mmr_iden_stack_5_point.toml -w 0 -a gpu
	```


### mmr_act - Action classification

* X: Point cloud (n, 3), Y: The action id. 

* Train command example:
	```bash
	python mm train mmr_act dgcnn --save mmr_act_dgcnn -config ./configs/act/mmr_act_stack_5_point.toml -w 0 -a gpu -m 300 -n -1
	```

* Test command example:
	```bash
	python mm test mmr_act dgcnn --load mmr_act_dgcnn -config ./configs/act/mmr_act_stack_5_point.toml -w 0 -a gpu
	```

## Ciation

If you find this work useful, please consider citing our paper:

```
@misc{cui2023milipoint,
      title={MiliPoint: A Point Cloud Dataset for mmWave Radar}, 
      author={Han Cui and Shu Zhong and Jiacheng Wu and Zichao Shen and Naim Dahnoun and Yiren Zhao},
      year={2023},
      eprint={2309.13425},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
