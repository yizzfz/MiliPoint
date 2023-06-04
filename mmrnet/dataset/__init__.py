import torch_geometric.transforms as T
import logging
import numpy as np

from .mmrnet_data import MMRKeypointData, MMRIdentificationData, MMRActionData
from torch_geometric.loader import DataLoader

dataset_map = {
    'mmr_kp': MMRKeypointData,
    'mmr_iden': MMRIdentificationData,
    'mmr_act': MMRActionData,
}

class Scale(T.BaseTransform):
    def __init__(self, factor) -> None:
        self.s = factor
        super().__init__()

    def __call__(self, data):
        x, y = data
        return x*self.s, y*self.s


transform_map = {
    'mmr_kp': (None, Scale(100)),
}

def get_dataset(name, batch_size, workers, mmr_dataset_config=None):
    dataset_cls = dataset_map[name]
    pre_transform, transform = transform_map.get(name, (None, None))
    if name in ['mmr_kp', 'mmr_iden', 'mmr_act']:
        train_dataset = dataset_cls(
            root=f'data/{name}', 
            partition='train', 
            transform=transform, 
            pre_transform=pre_transform, 
            mmr_dataset_config=mmr_dataset_config)
        val_dataset = dataset_cls(
            root=f'data/{name}', 
            partition='val', 
            transform=transform, 
            pre_transform=pre_transform, 
            mmr_dataset_config=mmr_dataset_config)
        test_dataset = dataset_cls(
            root=f'data/{name}', 
            partition='test', 
            transform=transform, 
            pre_transform=pre_transform, 
            mmr_dataset_config=mmr_dataset_config)
    else:
        train_dataset = dataset_cls(
            root=f'data/{name}', partition='train', transform=transform, pre_transform=pre_transform)
        val_dataset = dataset_cls(
            root=f'data/{name}', partition='val', transform=transform, pre_transform=pre_transform)
        test_dataset = dataset_cls(
            root=f'data/{name}', partition='test', transform=transform, pre_transform=pre_transform)
    if train_dataset.info['num_classes'] is not None:
        logging.info('Number of classes: %s' % train_dataset.info['num_classes'])
    else:
        logging.info('Number of keypoints: %s' % train_dataset.info['num_keypoints'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader, train_dataset.info