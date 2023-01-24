from argparse import ArgumentParser
import os
import sys
import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils import Params, seed_worker
from models.metric_learning import MetricLearningModel

def get_option():
    argparser = ArgumentParser(description='Training the network')
    argparser.add_argument('-p', '--param_file', type=str, default='hoge.json', help='filename of the parameter JSON')
    args = argparser.parse_args()
    return args

args = get_option()
print('parsed args :')
print(args)
params = Params(f'./params/{args.param_file}')
params.targets = None # Disable the regression mode.

# Reproducibility
seed=123
pl.seed_everything(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if params.encoder_name == "pointnet":
    from dataloaders.dataset_pointnet import MultimodalDatasetMP_PointNet as Dataset
elif params.encoder_name == "cgcnn":
    from dataloaders.dataset_cgcnn import MultimodalDatasetMP_CGCNN as Dataset
elif params.encoder_name.startswith("periodic"):
    from dataloaders.dataset_periodic import MultimodalDatasetMP_Periodic as Dataset
else:
    raise NameError(params.encoder_name)

# Setup datasets
if params.train_mode == 'train_val':
    train_dataset = Dataset(target_data='train', params=params)
    val_dataset = Dataset(target_data='val', params=params)
    test_dataset = Dataset(target_data='test', params=params)
elif params.train_mode == 'train_on_all':
    train_dataset = Dataset(target_data='train_on_all', params=params)
    val_dataset = train_dataset
    test_dataset = train_dataset

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True, 
                          worker_init_fn=seed_worker)
val_loader  = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4, drop_last=False)
# Uncomment below to force the updating of the cache files.
# train_dataset.process()
# val_dataset.process()
# test_dataset.process()

# Setup model and trainer
logger = loggers.TensorBoardLogger(params.save_path, name=params.experiment_name)
logger.log_hyperparams(params.__dict__)

if params.train_mode == 'train_on_all':
    checkpoint_callback = ModelCheckpoint(save_top_k=params.model_checkpoint_save_top_k,
        monitor='val/top01', mode='max', save_last=True,
        dirpath=logger.log_dir+'/model_checkpoint', every_n_epochs=50)
else:
    checkpoint_callback = ModelCheckpoint(save_top_k=params.model_checkpoint_save_top_k,
        monitor='val/top01', mode='max', save_last=True,
        dirpath=logger.log_dir+'/model_checkpoint')
        
        
trainer = Trainer(logger=logger, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1, auto_select_gpus=False, 
                    max_epochs=params.n_epochs,
                    default_root_dir=params.save_path,
                    callbacks=checkpoint_callback,
                    log_every_n_steps=10,
                    num_nodes=1,
                    limit_train_batches=params.train_percent_check,
                    limit_val_batches=params.val_percent_check,
                    fast_dev_run=False,
                    deterministic=False) # Change this true to reproduce the results
system = MetricLearningModel(params, train_loader, val_loader)

if params.pretrained_model is not None:
    system = system.load_from_checkpoint(
        params.pretrained_model,
        params=params,
        train_loader=train_loader,
        val_loader=val_loader,
        strict=False)

# Train model
trainer.fit(system)

# Prepare the best model for testing
if os.path.exists(checkpoint_callback.best_model_path):
    shutil.copyfile(checkpoint_callback.best_model_path, logger.log_dir+'/model_checkpoint/best.ckpt')
    system.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        params=params,
        train_loader=train_loader,
        val_loader=val_loader)

trainer.test(model=system, dataloaders=test_loader)

logger.finalize('success')  # to properly output all test scores in a TB log.

