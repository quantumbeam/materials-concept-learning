from numpy.lib import histograms
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.utils import normalize_embedding

def xrd_features(dropout_rate_conv=0.3):
    return nn.Sequential(
            # 9000→2001
            nn.Conv1d(in_channels=1, out_channels=80, kernel_size=100,
                      stride=5, padding=550),
            # BN入れるならこの行の位置
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_conv),

            # 2001→1000
            nn.AvgPool1d(kernel_size=3, stride=2, padding=0),

            # 1000→200
            nn.Conv1d(in_channels=80, out_channels=80, kernel_size=50,
                      stride=5, padding=23),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_conv),

            # 200→66
            nn.AvgPool1d(kernel_size=3, stride=None),

            # 66→33
            nn.Conv1d(in_channels=80, out_channels=80, kernel_size=25,
                      stride=2, padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_conv),
        )

def xrd_features_BN_9k(dropout_rate_conv=0.3):
    import torch.nn as nn
    return nn.Sequential(
            # 9000→2001
            nn.Conv1d(in_channels=1, out_channels=80, kernel_size=100,
                      stride=5, padding=550),
            nn.BatchNorm1d(80), #こんな感じで。todo; 引数を調べる
            nn.ReLU(inplace=True),

            # 2001→1000
            nn.AvgPool1d(kernel_size=3, stride=2, padding=0),

            # 1000→200
            nn.Conv1d(in_channels=80, out_channels=80, kernel_size=50,
                      stride=5, padding=23),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),

            # 200→66
            nn.AvgPool1d(kernel_size=3, stride=None),

            # 66→33
            nn.Conv1d(in_channels=80, out_channels=80, kernel_size=25,
                      stride=2, padding=12),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
        )

# XRD 2theta 10-110deg（5000 points）用にCNNを微調整
def xrd_features_BN_5k(dropout_rate_conv=0.3):
    import torch.nn as nn
    return nn.Sequential(
            # 5000→995
            nn.Conv1d(in_channels=1, out_channels=80, kernel_size=50,
                      stride=5, padding=10),
            nn.BatchNorm1d(80), #こんな感じで。todo; 引数を調べる
            nn.ReLU(inplace=True),

            # 995→200
            nn.Conv1d(in_channels=80, out_channels=80, kernel_size=10,
                      stride=5, padding=5),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),

            # 200→99
            nn.AvgPool1d(kernel_size=4, padding=0, stride=2),

            # 99→33
            nn.Conv1d(in_channels=80, out_channels=80, kernel_size=3,
                      stride=3, padding=0),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
        )


class XRD_CNN(torch.nn.Module):
    """
    xrd_encoder: str
        XRDのデータポイント数に合わせてXRDのエンコーダ（1D CNN）の設定を切り替える。
        xrd_features_BN_9kは9000点、〜_5kは5000点
    """
    def __init__(self, params, output_intermediate_feat=False):
        super(XRD_CNN, self).__init__()
        embedding_dim = params.embedding_dim
        self.params = params
        self.output_intermediate_feat = output_intermediate_feat

        # XRD CNNの部分
        self.xrd_features = xrd_features_BN_5k()

        # convの後
        self.xrd_avgpool = torch.nn.AvgPool1d(kernel_size=3, stride=None) # 880

        # TODO: ここのノード数は要検討
        self.xrd_lin1 = nn.Linear(880, embedding_dim)
        self.xrd_bn1 = nn.BatchNorm1d(num_features=embedding_dim)
        self.xrd_lin2 = nn.Linear(embedding_dim, embedding_dim)
        self.xrd_bn2 = nn.BatchNorm1d(num_features=embedding_dim)
        
        if not hasattr(params, 'targets') or params.targets is None:
            # embedding mode when targets = None
            self.xrd_lin3 = nn.Linear(embedding_dim, embedding_dim)
        else:
            # regression mode when targets = "hoge" or ["hoge", "foo"]
            final_dim = 1 if isinstance(params.targets, str) \
                else len(params.targets)
            self.xrd_regression = nn.Linear(embedding_dim, final_dim)


    def forward(self, data):
        if self.params.scale_xrd > 0:
            data.xrd = data.xrd / self.params.scale_xrd
        else:
            data.xrd = data.xrd / data.xrd.max(dim=2, keepdims=True)[0]

        # XRD CNNの出力
        output_xrd = self.xrd_features(data.xrd)
        output_xrd = self.xrd_avgpool(output_xrd)
        output_xrd = output_xrd.view(-1, 80 * 11)
        # for encoder
        if self.output_intermediate_feat:
          out1 = F.relu(self.xrd_bn1(self.xrd_lin1(output_xrd)))
          out2 = F.relu(self.xrd_bn2(self.xrd_lin2(out1)))
          return out1, out2

        output_xrd = self.xrd_lin1(output_xrd)
        output_xrd = F.relu(self.xrd_bn1(output_xrd))
        output_xrd = self.xrd_lin2(output_xrd)
        output_xrd = F.relu(self.xrd_bn2(output_xrd))

        if hasattr(self, 'xrd_regression'):
            return self.xrd_regression(output_xrd)

        output_xrd = self.xrd_lin3(output_xrd)
        output_xrd = normalize_embedding(output_xrd, self.params.embedding_normalize)
        return output_xrd

