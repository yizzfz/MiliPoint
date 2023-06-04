import sys
import random
import argparse
import functools
import logging

import torch
import numpy as np
import toml
import optuna

from .session import train, test
from .models import model_map 
from .dataset import get_dataset

logging.getLogger().setLevel(logging.INFO)

class Main:
    arguments = {
        ('action', ): {'type': str, 'help': 'Name of the action to perform.'},
        ('dataset', ): {'type': str, 'help': 'Name of the dataset.'},
        ('model', ): {'type': str, 'help': 'Name of the model.'},

        # checkpoint
        ('-load', '--load-name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to restore.'
        },
        ('-save', '--save-name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to save.'
        },

        # common training args
        ('-opt', '--optimizer'): {
            'type': str, 'default': 'adam', 'help': 'Pick an optimizer.',
        },
        ('-lr', '--learning-rate'): {
            'type': float, 'default': 1e-5, 'help': 'Initial learning rate.',
        },
        ('-m', '--max-epochs'): {
            'type': int, 'default': 100,
            'help': 'Maximum number of epochs for training.',
        },
        ('-b', '--batch-size'): {
            'type': int, 'default': 128,
            'help': 'Batch size for training and evaluation.',
        },
        ('-wd', '--weight-decay'): {
            'type': float, 'default': 1e-5, 'help': 'Weight decay for optimizer regularization.'
        },

        # debug control
        ('-d', '--debug'): {
            'action': 'store_true', 'help': 'Verbose debug',
        },
        ('-seed', '--seed'): {
            'type': int, 'default': 0, 'help': 'Number of steps for model optimisation',
        },

        # cpu gpu setup for lightning
        ('-w', '--num_workers'): {
            # multiprocessing fail with too many works
            'type': int, 'default': 0, 'help': 'Number of CPU workers.',
        },
        ('-n', '--num_devices'): {
            'type': int, 'default': 1, 'help': 'Number of GPU devices.',
        },
        ('-a', '--accelerator'): {
            'type': str, 'default': None, 'help': 'Accelerator style.',
        },
        ('-s', '--strategy'): {
            'type': str, 'default': None, 'help': 'Strategy style, e.g. ddp.',
        },
        # dataset related
        ('-config', '--dataset_config'): {
            'type': str, 'default': None, 'help': 'Dataset config.',
        },
        ('-d_seed', '--dataset_seed'): {
            'type': int, 'default': None, 'help': 'Dataset seed.',
        },
        ('-d_train_split', '--dataset_train_split'): {
            'type': float, 'default': None, 'help': 'Train dataset split.',
        },
        ('-d_val_split', '--dataset_val_split'): {
            'type': float, 'default': None, 'help': 'Val dataset split.',
        },
        ('-d_test_split', '--dataset_test_split'): {
            'type': float, 'default': None, 'help': 'Test dataset split.',
        },
        ('-d_stacks', '--dataset_stacks'): {
            'type': int, 'default': None, 'help': 'Number of mmPoints to stack.',
        },
        ('-d_zero_padding', '--dataset_zero_padding'): {
            'type': str, 'default': None, 'help': 'Zero padding styles.',
        },
        ('-d_max_points', '--dataset_max_points'): {
            'type': int, 'default': None, 'help': 'Point cloud population.',
        },
        ('-d_num_keypoints', '--dataset_num_keypoints'): {
            'type': str, 'default': 'low', 'help': 'Control the complexity of the human skeleton in the keypoint dataset.\
                                                    Supported values: \'low\' or \'high\'.',
        },
        ('-d_processed_data', '--dataset_processed_data'): {
            'type': str, 'default': None, 'help': 'Processed data file.',
        },
        ('-d_forced_rewrite', '--dataset_forced_rewrite'): {
            'action': 'store_true', 'help': 'Force to rewrite the processed data.',
        },
        ('-v', '--visualize'): {
            'action': 'store_true', 'help': 'Visualize test result as mp4.',
        },
        # tuner args
        ('-n_trials', '--n_trials'): {
            'type': int, 'default': 100, 'help': 'Number of trails to run.',
        },

    }

    def __init__(self):
        super().__init__()
        a = self.parse()
        if a.debug:
            sys.excepthook = self._excepthook
        # seeding
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        self.a = a

    def parse(self):
        p = argparse.ArgumentParser(description='Millimeter Wave Radar Dataset.')
        for k, v in self.arguments.items():
            p.add_argument(*k, **v)
        p = p.parse_args()
        p = self.post_parse(p)
        return p
    
    def post_parse(self, p):
        # load from a toml config file
        if p.dataset_config is not None:
            if not p.dataset_config.endswith('.toml'):
                raise ValueError('Dataset config must be a Toml file.')
            with open(p.dataset_config, 'r') as f:
                config = toml.load(f)
            
            cli_override_flag = False
            for k, v in config.items():
                k = f'dataset_{k}'
                if k in p:
                    cli_v = getattr(p, k)
                    if cli_v is None:
                        setattr(p, k, v)
                    else:
                        print(
                            f'[Dataset config setup] Config value {v} is not used for {k}',
                            f'command line has a higher priority and sets it to {cli_v}')
                        cli_override_flag = True
                else:
                    raise ValueError(f'Unknown config key {k}.')
            if cli_override_flag:
                if p.dataset_processed_data is None:
                    raise ValueError(
                        'We found you are overriding config file!',
                        'Please specify the processed data file path with -d_processed_data.'
                        'This is to avoid accidentally overwriting the processed data file.')
        return p

    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb
        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in [KeyboardInterrupt, FileNotFoundError]:
            if issubclass(etype, exc):
                sys.exit(-1)
        import ipdb
        ipdb.post_mortem(etb)

    def run(self):
        try:
            action = getattr(self, f'cli_{self.a.action.replace("-", "_")}')
        except AttributeError:
            callables = [n[4:] for n in dir(self) if n.startswith('cli_')]
            logging.error(
                f'Unkown action {self.a.action!r}, '
                f'accepts: {", ".join(callables)}.')
        return action()

    def setup_model_and_data(self, a, dataset_custom_args=None):
        # get dataset
        logging.info(f'Loading dataset {a.dataset!r}...')

        dataset_stacks = None if dataset_custom_args is None else dataset_custom_args.get('stacks', None)
        my_stacks = a.dataset_stacks if dataset_stacks is None else dataset_stacks

        path = None
        if dataset_custom_args is not None:
            path = a.dataset_processed_data.split("/")
            path[-1] = f"stack_{my_stacks}.pkl"
            path = ("/").join(path)

        dataset_num_keypoints = {
            'low': 9,
            'high': 18,
        }
        mmr_dataset_config = {
            'seed': a.dataset_seed,
            'train_split': a.dataset_train_split,
            'val_split': a.dataset_val_split,
            'test_split': a.dataset_test_split,
            'stacks': my_stacks,
            'zero_padding': a.dataset_zero_padding,
            'processed_data': a.dataset_processed_data if path is None else path,
            'forced_rewrite': a.dataset_forced_rewrite,
            'max_points': a.dataset_max_points,
            'num_keypoints': dataset_num_keypoints[a.dataset_num_keypoints],
        }
        train_loader, val_loader, test_loader, dataset_info = get_dataset(
            name=a.dataset, 
            batch_size=a.batch_size, 
            workers=a.num_workers,
            mmr_dataset_config=mmr_dataset_config)
        logging.info(f'Loaded dataset {a.dataset!r}.')

        # get model
        model_cls = model_map[a.model]
        model = model_cls(info=dataset_info)
        return model, train_loader, val_loader, test_loader

    def cli_train(self, dataset_custom_args=None, train_custom_args=None):
        a = self.a
        if not a.save_name:
            logging.error('--save-name not specified.')
            sys.exit(1)

        model, train_loader, val_loader, test_loader = self.setup_model_and_data(
            a, dataset_custom_args=dataset_custom_args)
        if a.strategy == 'ddp':
            a.strategy = 'ddp_find_unused_parameters_false'
        plt_trainer_args = {
            'max_epochs': a.max_epochs, 'devices': a.num_devices,
            'accelerator': a.accelerator, 'strategy': a.strategy,
            'fast_dev_run': a.debug,}
        
        optimizer = a.optimizer if train_custom_args is None else train_custom_args.get('optimizer', a.optimizer)
        learning_rate = a.learning_rate if train_custom_args is None else train_custom_args.get('learning_rate', a.learning_rate)
        weight_decay = a.weight_decay if train_custom_args is None else train_custom_args.get('weight_decay', a.weight_decay)
        
        train_params = {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "weight_decay": weight_decay,
            "save_path":  'checkpoints/' +a.save_name,
        }
        loss = train(**train_params)
        return loss

    def cli_test(self, dataset_custom_args=None):
        a = self.a

        model, train_loader, val_loader, test_loader = self.setup_model_and_data(
            a, dataset_custom_args=dataset_custom_args)
        
        load_path = a.load_name if a.load_name.endswith(".ckpt") else 'checkpoints/' + a.load_name + '/'

        plt_trainer_args = {
            'devices': a.num_devices,
            'accelerator': a.accelerator, 'strategy': a.strategy,}
        test_params = {
            'model': model,
            'test_loader': test_loader,
            'plt_trainer_args': plt_trainer_args,
            'load_path': load_path,
            'visualize': a.visualize,
        }
        test(**test_params)
    cli_eval = cli_test

    def cli_tune(self):
        def objective(trial):
            # define hyperparameter search space
            stack = trial.suggest_categorical("stack", [1, 3, 5, 25, 50, 75])
            learning_rate = trial.suggest_categorical("learning_rate", [1e-6, 1e-5, 2e-5, 5e-5, 1e-4, 1e-3, 1e-2])
            weight_decay = trial.suggest_categorical("weight_decay", [0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])

            val_loss = self.cli_train(
                dataset_custom_args={'stacks': stack},
                train_custom_args={'learning_rate': learning_rate, 'weight_decay': weight_decay})
            return val_loss
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.a.n_trials)
        print(study.best_trial)


def main():
    Main().run()
