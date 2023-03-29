import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_lightning.core.module import LightningModule
import torch.optim as optim

from models.pointnet import CrystalEncoder
from models.xrd_cnn import XRD_CNN
from models.cgcnn import CGCNN
from models.periodic_pointnet import PeriodicPointNet
from losses.triplet_loss import multimodal_triplet_loss
from utils import calc_grobal_top_k_acc


def batch_wise_accuracy(x, y):
    """
    Within each mini-batch, this function performs cross-modal nearest neighbor search for each XRD embedding and crystal embedding, and evaluates the retrieval accuracy.
    If the embedding of the crystal corresponding to the queried XRD embedding is the nearest neighbor, it is considered as correct. 
    To do this, this function counts the rows in which the diagonal component of the distance matrix is the minimum in row-wise.

    Example:
    pairwise_dist_mat =
    tensor([[10.2470, 12.2066,  6.3246,  7.6811,  7.2801],
            [ 9.0000,  8.0623,  6.6332, 11.0905,  6.0828],
            [ 2.4495,  3.7417,  7.6811,  4.8990, 10.2956],
            [ 7.0711,  7.2111,  2.2361,  5.0990,  9.4868],
            [ 5.3852,  7.2801,  8.8318,  8.7750,  4.1231]])
    In case of this, the result is:
    batch_wise_pair_correct = tensor([0, 0, 0, 0, 1], dtype=torch.uint8)

    Parameters
    ----------
    x : torch.tensor (xrd embedding)
    y : torch.tensor (crystal embedding)

    Returns
    -------
    batch_wise_pair_correct: int
    batch_wise_pair_accuracy : float
    """

    pairwise_dist_mat = torch.cdist(x, y, p=2)
    positive_dist = pairwise_dist_mat.diag()
    n_sample = pairwise_dist_mat.shape[0]
    inf_arr = torch.ones(n_sample).type_as(x) * float("Inf")
    # Replace the distance of the diagonal components (=pair XRD and crystal) in the distance matrix with inf
    negative_pairwise_dist_mat = pairwise_dist_mat + torch.diag(inf_arr)
    # Calculate the percentage of rows for which the diagonal components of the distance matrix are row-wise minimums.
    batch_wise_pair_correct = positive_dist < negative_pairwise_dist_mat.min(1)[0]
    batch_wise_pair_accuracy = torch.sum(batch_wise_pair_correct).float()/n_sample
    return batch_wise_pair_correct, batch_wise_pair_accuracy


class MetricLearningModel(LightningModule):
    """A class representing a metric learning model using LightningModule.

    Attributes:
    model (CrystalEncoder): The crystal encoder model specified by the encoder_name.
    model_xrd (XRD_CNN): The CNN model used for XRD data.
    train_loader: A dataloader containing the training data.
    val_loader: A dataloader containing the validation data.
    params: Model parameters such as learning rate, encoder name etc.
    """
    def __init__(self, params, train_loader, val_loader):
        super(MetricLearningModel, self).__init__()

        if params.encoder_name == "pointnet":
            self.model = CrystalEncoder(params)
        elif params.encoder_name == "cgcnn":
            self.model = CGCNN(params)
        elif params.encoder_name.startswith('periodic'):
            self.model = PeriodicPointNet(params)
        else:
            raise Exception(f"Invalid params.encoder_name: {params.encoder_name}")

        self.model_xrd = XRD_CNN(params)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params


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
        output_cry = self.model(batch)
        output_xrd = self.model_xrd(batch)

        loss = multimodal_triplet_loss(output_xrd, output_cry, self.params)
        _, batch_acc = batch_wise_accuracy(output_xrd, output_cry)

        output = {
            'loss': loss,
            'progress_bar': {'tr/loss': loss, 'tr/acc':batch_acc},
            'log': {'train/loss': loss, 'train/batch_acc':batch_acc}
        }
        self.log('train/loss', loss)
        self.log('train/batch_acc', batch_acc)
        return output

    def validation_step(self, batch, batch_idx):
        out_cry = self.model(batch)
        out_xrd = self.model_xrd(batch)

        loss = multimodal_triplet_loss(out_xrd, out_cry, self.params)
        _, batch_acc = batch_wise_accuracy(out_xrd, out_cry)
        
        return {
            'val/loss': loss, 
            'val/acc': batch_acc.float(),
            'out_cry': out_cry.detach().cpu(),
            'out_xrd': out_xrd.detach().cpu(),
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        acc = torch.stack([x['val/acc'] for x in outputs]).mean()
        cry = torch.cat([x['out_cry'] for x in outputs], dim=0)
        xrd = torch.cat([x['out_xrd'] for x in outputs], dim=0)
        topk_acc = calc_grobal_top_k_acc(embedding_query=xrd, embedding_target=cry, k=10)
        logs = {'val/loss': avg_loss, 'val/acc': acc}
        
        self.log('val/loss', avg_loss)
        self.log('val/acc', acc)
        print(f'\r\rval acc: {acc*100:4.1f} ', end='')
        for i in range(len(topk_acc)):
            logs['val/top%02d' % (i+1)] = torch.tensor(topk_acc[i])
            self.log('val/top%02d' % (i+1), torch.tensor(topk_acc[i]))
            print(f'top{i+1}: {topk_acc[i]*100:4.1f} ', end='')
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
            self.log(newkey, val_log[key])
        return {'log': test_log}

    def configure_optimizers(self):
        p = []
        p.extend(self.model.parameters())
        p.extend(self.model_xrd.parameters())
        return optim.Adam(p, lr=self.params.lr)
        
    def forward(self, x):
        output_cry = self.model(x)
        output_xrd = self.model_xrd(x)
        return output_cry, output_xrd

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
