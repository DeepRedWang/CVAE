import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy.io import loadmat
device = "cpu"
# "cuda:2" if torch.cuda.is_available() else
class GetTestData(Dataset):
    """
    Return data for the dataloader.
    """
    def __init__(self, name='Yak42_data64_256'):
        filename = name.split('_')[0]
        Y = loadmat(f'./Data/Test/Dfixed64_256/{filename}/{name}.mat')
        self.Y = torch.tensor(Y[f'{name}'], dtype=torch.complex64, device=device)
        self.length = len(self.Y)

    def __getitem__(self, item):
        data = {}
        data['test_Y'] = self.Y[item]
        return data

    def __len__(self):
        return self.length

class GetTestData1(Dataset):
    """
    Return data for the dataloader.
    """
    def __init__(self, name='Ironplane_data64_256'):
        # filename = name.split('_')[0]
        filename = 'Ironplane'
        Y = loadmat(f'./Data/Test/Dfixed64_256/{filename}/{name}.mat')
        self.Y = torch.tensor(Y[f'{name}'], dtype=torch.complex64, device=device)
        self.length = len(self.Y)

    def __getitem__(self, item):
        data = {}
        data['test_Y'] = self.Y[item]
        return data

    def __len__(self):
        return self.length

class GetTestData32(Dataset):
    """
    Return data for the dataloader.
    """
    def __init__(self, name='Ironplane_data64_256'):
        # filename = name.split('_')[0]
        filename = 'Ironplane'
        Y = loadmat(f'./Data/Test/Dfixed32_256/{filename}/{name}.mat')
        self.Y = torch.tensor(Y[f'{name}'], dtype=torch.complex64, device=device)
        self.length = len(self.Y)

    def __getitem__(self, item):
        data = {}
        data['test_Y'] = self.Y[item]
        return data

    def __len__(self):
        return self.length


# class GetTestData1(Dataset):
#     """
#     Return data for the dataloader.
#     """
#     def __init__(self):
#         # R_y = loadmat('E:/WangJianyang/MatLabPro/Sparse Aperture/test/D/ship1/test_Y_real.mat')
#         # I_y = loadmat('E:/WangJianyang/MatLabPro/Sparse Aperture/test/D/ship1/test_Y_imag.mat')
#         R_y = loadmat('./Data/test/D/ship1/test_Y_real.mat')
#         I_y = loadmat('./Data/test/D/ship1/test_Y_imag.mat')
#         self.R_y = self.trans(R_y['test_Y_real']).squeeze().to(dtype=torch.float32, device=device)
#         self.I_y = self.trans(I_y['test_Y_imag']).squeeze().to(dtype=torch.float32, device=device)
#         self.length = len(self.R_y)
#
#     def __getitem__(self, item):
#         data = {}
#         data['test_Y_real'] = self.R_y[item]
#         data['test_Y_imag'] = self.I_y[item]
#         return data
#     # {'Y_real':self.R_y[item], 'Y_imag':self.I_y[item], 'X_real':self.R_x[item], 'X_imag':self.I_x[item]}
#
#     def __len__(self):
#         return self.length
#
#     def trans(self, data):
#         trans = transforms.Compose([
#             transforms.ToTensor()
#         ])
#         return trans(data)