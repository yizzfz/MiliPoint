from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from .wrapper import ModelWrapper


def train(
        model,
        train_loader, val_loader,
        optimizer, 
        learning_rate, 
        weight_decay,
        plt_trainer_args, 
        save_path):
    plt_model = ModelWrapper(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=plt_trainer_args['max_epochs'],
        optimizer=optimizer)
    metric = f'val_{plt_model.metric_name}'
    if 'mle' in metric:
        mode = 'min'
    elif 'acc' in metric:
        mode = 'max'
    else:
        raise ValueError(f'Unknown metric {metric}')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric,
        mode=mode,
        filename="best",
        dirpath=save_path,
        save_last=True,
    )
    plt_trainer_args['callbacks'] = [checkpoint_callback]
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.fit(plt_model, train_loader, val_loader)
    return plt_model.best_val_loss
