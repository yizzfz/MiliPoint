import os
import torch
import numpy as np
import pickle
import logging
import random
from tqdm import tqdm
import json

from torch_geometric.data import Dataset
from torch_geometric.data.collate import collate


class MMRKeypointData(Dataset):
    raw_data_path = 'data/raw'
    processed_data = 'data/processed/mmr_kp/data.pkl'
    max_points = 22
    seed = 42
    partitions = (0.8, 0.1, 0.1)
    stacks = None
    zero_padding = 'per_data_point'
    zero_padding_styles = ['per_data_point', 'per_stack', 'data_point', 'stack']
    num_keypoints = 9
    forced_rewrite = False

    def _parse_config(self, c):
        c = {k: v for k, v in c.items() if v is not None}
        self.seed = c.get('seed', self.seed)
        self.processed_data = c.get('processed_data', self.processed_data)
        self.max_points = c.get('max_points', self.max_points)
        self.partitions = (
            c.get('train_split', self.partitions[0]),
            c.get('val_split', self.partitions[1]),
            c.get('test_split', self.partitions[2]))
        self.stacks = c.get('stacks', self.stacks)
        self.zero_padding = c.get('zero_padding', self.zero_padding)
        self.num_keypoints = c.get('num_keypoints', self.num_keypoints)
        if self.zero_padding not in self.zero_padding_styles:
            raise ValueError(
                f'Zero padding style {self.zero_padding} not supported.')
        self.forced_rewrite = c.get('forced_rewrite', self.forced_rewrite)

    def __init__(
            self, root, partition, 
            transform=None, pre_transform=None, pre_filter=None,
            mmr_dataset_config = None):
        super(MMRKeypointData, self).__init__(
            root, transform, pre_transform, pre_filter)
        self._parse_config(mmr_dataset_config)
        # check if processed_data exists
        if (not os.path.isfile(self.processed_data)) or self.forced_rewrite:
            self.data, _ = self._process()
            with open(self.processed_data, 'wb') as f:
                pickle.dump(self.data, f)
        else:
            with open(self.processed_data, 'rb') as f:
                self.data = pickle.load(f)
        total_samples = len(self.data['train']) + len(self.data['val']) + len(self.data['test'])
        self.data = self.data[partition]
        self.num_samples = len(self.data)
        self.target_dtype = torch.float
        self.info = {
            'num_samples': self.num_samples,
            'num_keypoints': self.num_keypoints,
            'num_classes': None,
            'max_points': self.max_points,
            'stacks': self.stacks,
            'partition': partition,
        }
        logging.info(
            f'Loaded {partition} data with {self.num_samples} samples,'
            f' where the total number of samples is {total_samples}')

    def len(self):
        return self.num_samples
    
    def get(self, idx):
        data_point = self.data[idx]
        x = data_point['new_x']
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(data_point['y'], dtype=self.target_dtype)
        return x, y

    @property
    def raw_file_names(self):
        file_names = [i for i in range(19)]
        return [f'{self.raw_data_path}/{i}.pkl' for i in file_names]

    def _process(self):
        data_list = []
        for fn in self.raw_file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            data_list = data_list + data_slice
        num_samples = len(data_list)
        logging.info(f'Loaded {num_samples} data points')

        # stack and pad frames based on config
        data_list = self.transform_keypoints(data_list)
        data_list = self.stack_and_padd_frames(data_list)

        #random shuffle train and val data
        random.seed(self.seed)
        random.shuffle(data_list)

        # get partitions
        train_end = int(self.partitions[0] * num_samples)
        val_end = train_end + int(self.partitions[1] * num_samples)
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        data_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }
        return data_map, num_samples

    def stack_and_padd_frames(self, data_list):
        if self.stacks is None:
            return data_list
        # take multiple frames for each x
        xs = [d['x'] for d in data_list]
        stacked_xs = []
        padded_xs = []
        print("Stacking and padding frames...")
        pbar = tqdm(total=len(xs))

        if self.zero_padding in ['per_data_point', 'data_point']:
            for i in range(len(xs)):
                data_point = []
                for j in range(self.stacks):
                    if i - j >= 0:
                        mydata_slice = xs[i - j]
                        diff = self.max_points - mydata_slice.shape[0]
                        mydata_slice = np.pad(mydata_slice, ((0, max(diff, 0)), (0, 0)), 'constant')
                        mydata_slice = mydata_slice[np.random.choice(len(mydata_slice), self.max_points, replace=False)]  
                        data_point.append(mydata_slice)
                    else:
                        data_point.append(np.zeros((self.max_points, 3)))
                padded_xs.append(np.concatenate(data_point, axis=0))
                pbar.update(1)
        elif self.zero_padding in ['per_stack', 'stack']:
            for i in range(len(xs)):
                start = max(0, i - self.stacks)
                stacked_xs.append(np.concatenate(xs[start:i+1], axis=0))
                pbar.update(0.5)
            for x in stacked_xs:
                diff = self.max_points * self.stacks - x.shape[0]
                x = np.pad(x, ((0, max(diff, 0)), (0, 0)), 'constant')
                x = x[np.random.choice(len(x), self.max_points * self.stacks, replace=False)]  
                padded_xs.append(x)
                pbar.update(0.5)
        else:
            raise NotImplementedError()
        pbar.close()
        print("Stacking and padding frames done")
        # remap padded_xs to data_list
        new_data_list = [{**d, 'new_x': x} for d, x in zip(data_list, padded_xs)]
        return new_data_list
    
    kp18_names = ['NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 
                  'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW', 
                  'LEFT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE', 
                  'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE', 
                  'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE', 
                  'RIGHT_EAR', 'LEFT_EAR']
    kp9_names = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 
                 'LEFT_SHOULDER', 'LEFT_ELBOW', 
                 'RIGHT_HIP', 'RIGHT_KNEE', 
                 'LEFT_HIP', 'LEFT_KNEE', 'HEAD']
    head_names = ['NOSE', 'RIGHT_EYE', 'LEFT_EYE', 'RIGHT_EAR', 'LEFT_EAR']
    def transform_keypoints(self, data_list):
        if self.num_keypoints == 18:
            return data_list
        
        print("Transforming keypoints ...")
        self.kp9_idx = [self.kp18_names.index(n) for n in self.kp9_names[:-1]]
        self.head_idx = [self.kp18_names.index(n) for n in self.head_names]
        for data in tqdm(data_list, total=len(data_list)):
            kpts = data['y']
            kpts_new = kpts[self.kp9_idx]
            head = np.mean(kpts[self.head_idx], axis=0)
            kpts_new = np.concatenate((kpts_new, head[None]))
            assert kpts_new.shape == (9, 3)
            data['y'] = kpts_new
        print("Transforming keypoints done")
        return data_list


class MMRIdentificationData(MMRKeypointData):
    processed_data = 'data/processed/mmr_iden/data.pkl'
    def __init__(self, *args, **kwargs):
        iden_dict = open('./data/raw/id.json', 'r').read()
        iden_dict = json.loads(iden_dict)
        self.iden_dict = iden_dict
        super().__init__(*args, **kwargs)
        self.info['num_classes'] = len(iden_dict)
        self.target_dtype = torch.int64

    def _process(self):
        data_list = []
        iden_dict = self.iden_dict
        for i in iden_dict:
            files = iden_dict[i]
            files = [f'{self.raw_data_path}/{k}.pkl' for k in files]
            for fn in files:
                logging.info(f'Loading {fn}')
                with open(fn, 'rb') as f:
                    data_slice = pickle.load(f)
                data_slice = [{'x': d['x'], 'y': int(i)} for d in data_slice]
                data_list = data_list + data_slice
        num_samples = len(data_list)
        logging.info(f'Loaded {num_samples} data points')

        data_list = self.stack_and_padd_frames(data_list)
        #random shuffle train and val data
        random.seed(self.seed)
        random.shuffle(data_list)

        # get partitions
        train_end = int(self.partitions[0] * num_samples)
        val_end = train_end + int(self.partitions[1] * num_samples)
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        data_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }
        return data_map, num_samples

class MMRActionData(MMRKeypointData):
    processed_data = 'data/processed/mmr_act/data.pkl'
    def __init__(self, *args, **kwargs):
        self.action_label = np.load('./data/raw/action_label.npy')
        super().__init__(*args, **kwargs)
        self.info['num_classes'] = len(np.unique(self.action_label))-1 # except -1
        self.target_dtype = torch.int64

    def _process(self):
        data_list = []
        for fn in self.raw_file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            data_list = data_list + data_slice

        for i, data in enumerate(data_list):
            data['y'] = self.action_label[i]
        data_list = [d for d in data_list if d['y']!=-1]

        data_list = self.stack_and_padd_frames(data_list)
        num_samples = len(data_list)
        logging.info(f'Loaded {num_samples} data points')

        # get partitions
        train_end = int(self.partitions[0] * num_samples)
        val_end = train_end + int(self.partitions[1] * num_samples)
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        # #random shuffle train and val data
        random.seed(self.seed)
        random.shuffle(train_data)
        random.shuffle(val_data)

        data_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }
        return data_map, num_samples
    
    def stack_and_padd_frames(self, data_list):
        if self.stacks is None:
            return data_list
        # take multiple frames for each x
        xs = [d['x'] for d in data_list]
        stacked_xs = []
        padded_xs = []
        print("Stacking and padding frames...")
        pbar = tqdm(total=len(xs))

        if self.zero_padding in ['per_data_point', 'data_point']:
            for i in range(len(xs)):
                data_point = []
                for j in range(self.stacks):
                    if i - j >= 0 and self.action_label[i] == self.action_label[i-j]:
                        mydata_slice = xs[i - j]
                        diff = self.max_points - mydata_slice.shape[0]
                        mydata_slice = np.pad(mydata_slice, ((0, max(diff, 0)), (0, 0)), 'constant')
                        mydata_slice = mydata_slice[np.random.choice(len(mydata_slice), self.max_points, replace=False)]  
                        data_point.append(mydata_slice)
                    else:
                        data_point.append(np.zeros((self.max_points, 3)))
                padded_xs.append(np.concatenate(data_point, axis=0))
                pbar.update(1)
        elif self.zero_padding in ['per_stack', 'stack']:
            for i in range(len(xs)):
                start = max(0, i - self.stacks)
                while self.action_label[i] != self.action_label[start]:
                    start = start + 1
                stacked_xs.append(np.concatenate(xs[start:i+1], axis=0))
                pbar.update(0.5)
            for x in stacked_xs:
                diff = self.max_points * self.stacks - x.shape[0]
                x = np.pad(x, ((0, max(diff, 0)), (0, 0)), 'constant')
                x = x[np.random.choice(len(x), self.max_points * self.stacks, replace=False)]  
                padded_xs.append(x)
                pbar.update(0.5)
        else:
            raise NotImplementedError()
        pbar.close()
        print("Stacking and padding frames done")
        # remap padded_xs to data_list
        new_data_list = [{**d, 'new_x': x} for d, x in zip(data_list, padded_xs)]
        return new_data_list
