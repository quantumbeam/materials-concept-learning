
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_lightning.core.module import LightningModule
import torch.optim as optim
from losses.regression_loss import regression_loss

from models.pointnet import CrystalEncoder
from models.xrd_cnn import XRD_CNN
from models.cgcnn import CGCNN
from models.periodic_pointnet import PeriodicPointNet

class RegressionModel(LightningModule):
    """
    A PyTorch Lightning module for regression tasks.

    Args:
        params: (argparse.Namespace) The configuration parameters.
        train_loader: (torch.utils.data.DataLoader) The data loader for training set.
        val_loader: (torch.utils.data.DataLoader)  The data loader for validation set. 
    """
    def __init__(self, params, train_loader, val_loader):
        super(RegressionModel, self).__init__()

        if params.encoder_name == "pointnet":
            self.model = CrystalEncoder(params)
        elif params.encoder_name == "cgcnn":
            self.model = CGCNN(params)
        elif params.encoder_name == "xrd":
            # Use model_xrd instead of model for loading state_dict.
            self.model_xrd = XRD_CNN(params)
        elif params.encoder_name.startswith('periodic'):
            self.model = PeriodicPointNet(params)
        else:
            raise Exception(f"Invalid params.encoder_name: {params.encoder_name}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.hparams = params.__dict__

        self.targets = self.params.targets
        if isinstance(self.targets, str):
            self.targets = [self.targets]

    def load_state_dict(self, state_dict, strict: bool = True):
        # Override for backward compatibility
        new_dict = {}
        for key in state_dict:
            if key.startswith("model.xrd_"):
                # replace 'model' with 'model_xrd'
                new_dict['model_xrd' + key[5:]] = state_dict[key]
            else:
                new_dict[key] = state_dict[key]

        return super().load_state_dict(new_dict, strict)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = regression_loss(output, batch, self.targets)

        output = {
            'loss': loss,
            'progress_bar': {"tr/loss": loss},
            'log': {'train/loss': loss}
        }
        return output

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = regression_loss(output, batch, self.targets)

        out = {
            'val/loss': loss, 
            'output': output.detach().cpu(),
        }

        for i, t in enumerate(self.targets):
            labels = batch[t]
            out[t] = abs(output[:, i] - labels).detach().cpu()

        return out

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        logs = {'val/loss': avg_loss}
        
        print(f'\r\rval loss: {avg_loss:.3f} ', end='')
        for t in self.targets:
            v = torch.cat([x[t] for x in outputs], dim=0).mean()
            logs['val/' + t] = v
            print(f'{t}: {v.item():.3f} ', end='')
        print('   ')

        return {'log': logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
        
    def test_epoch_end(self, outputs):
        out = self.validation_epoch_end(outputs)
        val_log = out['log']
        test_log = {}
        for key in val_log:
            newkey = key.replace('val/', 'test/')
            test_log[newkey] = val_log[key]
        return {'log': test_log}

    def configure_optimizers(self):
        if hasattr(self, 'model'):
            return optim.Adam(self.model.parameters(), lr=self.params.lr)
        
        return optim.Adam(self.model_xrd.parameters(), lr=self.params.lr)

    def forward(self, x):
        if hasattr(self, 'model'):
            return self.model(x)

        return self.model_xrd(x)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
