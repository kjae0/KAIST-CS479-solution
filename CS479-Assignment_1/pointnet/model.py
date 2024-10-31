import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.fc3x64 = nn.Linear(3, 64)
        self.bn64 = nn.BatchNorm1d(64)
        self.fc64x128 = nn.Linear(64, 128)
        self.bn128 = nn.BatchNorm1d(128)
        self.fc128x1024 = nn.Linear(128, 1024)
        self.bn1024 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        if self.input_transform:
            trans_mat = self.stn3(pointcloud.permute(0, 2, 1)) # B x 3 x 3
            pointcloud = torch.bmm(pointcloud, trans_mat) # B x N x 3
      
        pointcloud = self.fc3x64(pointcloud) # B x N x 64
        pointcloud = self.bn64(pointcloud.permute(0, 2, 1)).permute(0, 2, 1)
        pointcloud = self.relu(pointcloud)

        if self.feature_transform:
            trans_mat = self.stn64(pointcloud.permute(0, 2, 1)) # B x 64 x 64
            pointcloud = torch.bmm(pointcloud, trans_mat) # B x N x 64
            res = pointcloud.clone()
        else:
            res = None
        
        pointcloud = self.fc64x128(pointcloud) # B x N x 128
        pointcloud = self.bn128(pointcloud.permute(0, 2, 1)).permute(0, 2, 1)
        pointcloud = self.relu(pointcloud)
        pointcloud = self.fc128x1024(pointcloud) # B x N x 1024
        pointcloud = self.bn1024(pointcloud.permute(0, 2, 1)).permute(0, 2, 1)
        pointcloud = self.relu(pointcloud)
        
        pooled = torch.max(pointcloud, 1)[0] # B x 1024
        
        return pooled, res


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        feat, _ = self.pointnet_feat(pointcloud)
        out = self.fc(feat)
        
        return out
    

class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.pointnet_feat = PointNetFeat(input_transform=True,
                                          feature_transform=True)
        self.fc1088x512 = nn.Linear(1088, 512)
        self.bn512 = nn.BatchNorm1d(512)
        self.fc512x256 = nn.Linear(512, 256)
        self.bn256 = nn.BatchNorm1d(256)
        self.fc256x128 = nn.Linear(256, 128)
        self.bn128 = nn.BatchNorm1d(128)
        self.fc128xm = nn.Linear(128, m)
        self.relu = nn.ReLU()
        
        

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        # feat: B x 1024
        # res: B x N x 64
        feat, res = self.pointnet_feat(pointcloud)
        feat = torch.unsqueeze(feat, 2).repeat(1, 1, pointcloud.size(1))  # B x 1024 x N
        feat = feat.permute(0, 2, 1)  # B x N x 1024

        feat = torch.cat([feat, res], 2) # B x N x 1088

        feat = self.fc1088x512(feat) # B x N x 512
        feat = self.bn512(feat.permute(0, 2, 1)).permute(0, 2, 1)
        feat = self.relu(feat)
        feat = self.fc512x256(feat)
        feat = self.bn256(feat.permute(0, 2, 1)).permute(0, 2, 1)
        feat = self.relu(feat)
        feat = self.fc256x128(feat)
        feat = self.bn128(feat.permute(0, 2, 1)).permute(0, 2, 1)
        feat = self.relu(feat)  # B x N x 128
        
        logit = self.fc128xm(feat)  # B x N x m
        
        return logit.permute(0, 2, 1)  # B x m(50) x N


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.fc1024xn4 = nn.Linear(1024, num_points//4)
        self.bnn4 = nn.BatchNorm1d(num_points//4)
        self.fcn4xn2 = nn.Linear(num_points//4, num_points//2)
        self.bnn2 = nn.BatchNorm1d(num_points//2)
        self.fcn2xn = nn.Linear(num_points//2, num_points)
        self.dropout = nn.Dropout(p=0.1)
        self.bnn = nn.BatchNorm1d(num_points)
        self.fcnx3n = nn.Linear(num_points, num_points*3)
        self.relu = nn.ReLU()
        

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        feat, _ = self.pointnet_feat(pointcloud)  # B x 1024
        feat = self.fc1024xn4(feat)
        feat = self.bnn4(feat)
        feat = self.relu(feat)
        feat = self.fcn4xn2(feat)
        feat = self.bnn2(feat)
        feat = self.relu(feat)
        feat = self.fcn2xn(feat)
        feat = self.dropout(feat)
        feat = self.bnn(feat)
        feat = self.relu(feat)
        feat = self.fcnx3n(feat)  # B x N * 3
    
        return feat.reshape(-1, pointcloud.shape[1], 3)
        


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()


if __name__ == "__main__":
    x = torch.zeros(32, 12, 3)
    point_cls = PointNetCls(num_classes=17, input_transform=True, feature_transform=True)
    
    device = 'cuda:1'
    
    x = x.to(device)
    point_cls = point_cls.to(device)
    out = point_cls(x)
    print(out.shape)
    
    x = torch.zeros(32, 12, 3)
    point_seg = PointNetPartSeg(m=50)
    
    x = x.to(device)
    point_seg = point_seg.to(device)
    out = point_seg(x)
    print(out.shape)

    x = torch.zeros(32, 12, 3)
    point_ae = PointNetAutoEncoder(num_points=12)
    x = x.to(device)
    point_ae = point_ae.to(device)
    out = point_ae(x)
    print(out.shape)
