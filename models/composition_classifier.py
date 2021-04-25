import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositionClassifier(nn.Module):

    def __init__(self, input_dim, num_classes, normalization_sign=False):
        super().__init__()
        half_input_dim = int(input_dim / 2)

        self.mlp = nn.Linear(input_dim, half_input_dim)
        self.fc = nn.Linear(half_input_dim, num_classes)
        self.normalization = normalization_sign

    def forward(self, f1, f2):
        """
        :param f1: other modality (e.g. audio or vision)
        :param f2: video modality
        :return:
        """
        if self.normalization:
            f1_n = F.normalize(f1, dim=1)
            f2_n = F.normalize(f2, dim=1)
            residual = torch.cat((f1_n, f2_n), 1)
        else:
            residual = torch.cat((f1, f2), 1)

        #### compose two modalities by residual learning
        residual = self.mlp(residual) ### a simple MLP
        feature = f1 + residual # other modality + residual (default)

        ## perform classification here
        out = self.fc(feature)

        return out, feature

def generate_prediction(input_dim, num_classes, normalization):
    model = CompositionClassifier(input_dim=input_dim, num_classes=num_classes, normalization_sign=normalization)
    return model


