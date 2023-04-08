import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
device2 = "cuda:2" if torch.cuda.is_available() else "cpu"
device3 = "cuda:3" if torch.cuda.is_available() else "cpu"

class GetDatanew(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset='TrainRD32_256_500000', device=None):
        Y = h5py.File(f'Data/Train/{trainingset}/Y.mat')
        Y = np.transpose(Y['Y'])
        Y = Y['real'] + 1j*Y['imag']
        self.Y = torch.tensor(Y, dtype=torch.complex64, device=device)
        X = h5py.File(f'Data/Train/{trainingset}/X.mat')
        X = np.transpose(X['X'])
        X = X['real'] + 1j*X['imag']
        self.X = torch.tensor(X, dtype=torch.complex64, device=device)

    def __getitem__(self, item):
        data = {}
        data['Y'] = self.Y[item]
        data['X'] = self.X[item]
        return data

    def __len__(self):
        return len(self.Y)

class GetData(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset='TrainRD64_256_500000', device=device3):
        Y = h5py.File(f'Data/Train/{trainingset}/Y.mat')
        Y = np.transpose(Y['Y'])
        Y = Y['real'] + 1j*Y['imag']
        self.Y = torch.tensor(Y, dtype=torch.complex64, device=device)
        X = h5py.File(f'Data/Train/{trainingset}/X.mat')
        X = np.transpose(X['X'])
        X = X['real'] + 1j*X['imag']
        self.X = torch.tensor(X, dtype=torch.complex64, device=device)

    def __getitem__(self, item):
        data = {}
        data['Y'] = self.Y[item]
        data['X'] = self.X[item]
        return data

    def __len__(self):
        return len(self.Y)

# class GetData(Dataset):
#     """
#     return data for dataloader
#     """
#     def __init__(self):
#         Y = loadmat('E:\WangJianyang\MatLabPro\Sparse Aperture\TraintotalAzimuthnoise300000\Y.mat')['Y']
#         self.Y = torch.tensor(Y, dtype=torch.complex64, device=device)
#
#
#     def __getitem__(self, item):
#         data = {}
#         data['Y'] = self.Y[item]
#         return data
#
#     def __len__(self):
#         return len(self.Y)


class GetData1(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset='TrainRD32_256_500000'):
        Y = h5py.File(f'Data/Train/{trainingset}/Y.mat')
        Y = np.transpose(Y['Y'])
        Y = Y['real'] + 1j*Y['imag']
        self.Y = torch.tensor(Y, dtype=torch.complex64, device=device1)
        X = h5py.File(f'Data/Train/{trainingset}/X.mat')
        X = np.transpose(X['X'])
        X = X['real'] + 1j*X['imag']
        self.X = torch.tensor(X, dtype=torch.complex64, device=device1)

    def __getitem__(self, item):
        data = {}
        data['Y'] = self.Y[item]
        data['X'] = self.X[item]
        return data

    def __len__(self):
        return len(self.Y)

# class GetData1(Dataset):
#     """
#     return data for dataloader
#     """
#     def __init__(self):
#         Y = loadmat('E:\WangJianyang\MatLabPro\Sparse Aperture\TraintotalAzimuthnoise300000\Y.mat')['Y']
#         self.Y = torch.tensor(Y, dtype=torch.complex64, device=device1)
#
#     def __getitem__(self, item):
#         data = {}
#         data['Y'] = self.Y[item]
#         return data
#
#     def __len__(self):
#         return len(self.Y)


# class GetData2(Dataset):
#     """
#     return data for dataloader
#     """
#     def __init__(self, trainingset='TrainRD32_256_1000000'):
#         Y = h5py.File(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\Y.mat')
#         Y = np.transpose(Y['Y'])
#         Y = Y['real'] + 1j*Y['imag']
#         self.Y = torch.tensor(Y, dtype=torch.complex64, device=device2)
#         X = h5py.File(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\X.mat')
#         X = np.transpose(X['X'])
#         X = X['real'] + 1j*Y['imag']
#         self.X = torch.tensor(X, dtype=torch.complex64, device=device2)
#         # self.X = torch.tensor(X, dtype=torch.complex64, device=device2)
#         # print(mat['your_dataset_name'].shape)
#         # Y = loadmat(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\Y.mat')['Y']
#         # X = loadmat(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\X.mat')['X']
#
#         # self.Y = torch.tensor(Y, dtype=torch.complex64, device=device2)
#         # self.X = torch.tensor(X, dtype=torch.complex64, device=device2)
#
#     def __getitem__(self, item):
#         data = {}
#         data['Y'] = self.Y[item]
#         data['X'] = self.X[item]
#         return data
#
#     def __len__(self):
#         return len(self.Y)

class GetData2(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset='TrainRD32_256_500000'):
        Y = h5py.File(f'Data/Train/{trainingset}/Y.mat')
        Y = np.transpose(Y['Y'])
        Y = Y['real'] + 1j*Y['imag']
        # self.Y = torch.tensor(Y, dtype=torch.complex64, device=device2)
        self.Y = torch.tensor(Y, dtype=torch.complex64)
        X = h5py.File(f'Data/Train/{trainingset}/X.mat')
        X = np.transpose(X['X'])
        X = X['real'] + 1j*X['imag']
        # self.X = torch.tensor(X, dtype=torch.complex64, device=device2)
        self.X = torch.tensor(X, dtype=torch.complex64)


    def __getitem__(self, item):
        data = {}
        data['Y'] = self.Y[item].to(device2)
        data['X'] = self.X[item].to(device2)
        return data

    def __len__(self):
        return len(self.Y)


class GetData3(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingset='TrainRD64_256_500000'):
        Y = h5py.File(f'Data/Train/{trainingset}/Y.mat')
        Y = np.transpose(Y['Y'])
        Y = Y['real'] + 1j*Y['imag']
        self.Y = torch.tensor(Y, dtype=torch.complex64, device=device3)
        X = h5py.File(f'Data/Train/{trainingset}/X.mat')
        X = np.transpose(X['X'])
        X = X['real'] + 1j*X['imag']
        self.X = torch.tensor(X, dtype=torch.complex64, device=device3)

    def __getitem__(self, item):
        data = {}
        data['Y'] = self.Y[item]
        data['X'] = self.X[item]
        return data

    def __len__(self):
        return len(self.Y)



# class GetData3(Dataset):
#     """
#     return data for dataloader
#     """
#     def __init__(self, trainingset='TrainRD32_256_1000000'):
#         Y = h5py.File(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\Y.mat')
#         Y = np.transpose(Y['Y'])
#         Y = Y['real'] + 1j*Y['imag']
#         self.Y = torch.tensor(Y, dtype=torch.complex64, device=device3)
#         X = h5py.File(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\X.mat')
#         X = np.transpose(X['X'])
#         X = X['real'] + 1j*X['imag']
#         self.X = torch.tensor(X, dtype=torch.complex64, device=device3)
#         # self.X = torch.tensor(X, dtype=torch.complex64, device=device2)
#         # print(mat['your_dataset_name'].shape)
#         # Y = loadmat(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\Y.mat')['Y']
#         # X = loadmat(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\X.mat')['X']
#
#         # self.Y = torch.tensor(Y, dtype=torch.complex64, device=device2)
#         # self.X = torch.tensor(X, dtype=torch.complex64, device=device2)
#
#     def __getitem__(self, item):
#         data = {}
#         data['Y'] = self.Y[item]
#         data['X'] = self.X[item]
#         return data
#
#     def __len__(self):
#         return len(self.Y)


# class GetData3(Dataset):
#     """
#     return data for dataloader
#     """
#     def __init__(self, trainingset='TrainRD32_256_500000'):
#         Y = loadmat(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\Y.mat')['Y']
#         X = loadmat(f'E:\WangJianyang\MatLabPro\Sparse Aperture\{trainingset}\X.mat')['X']
#
#         self.Y = torch.tensor(Y, dtype=torch.complex64, device=device3)
#         self.X = torch.tensor(X, dtype=torch.complex64, device=device3)
#
#     def __getitem__(self, item):
#         data = {}
#         data['Y'] = self.Y[item]
#         data['X'] = self.X[item]
#         return data
#
#     def __len__(self):
#         return len(self.Y)


# class GetData2(Dataset):
#     """
#     return data for dataloader
#     """
#     def __init__(self):
#         R_y = loadmat('./Data/TrainAzimuth100000/Y_real.mat')
#         I_y = loadmat('./Data/TrainAzimuth100000/Y_imag.mat')
#         R_x = loadmat('./Data/TrainAzimuth100000/X_real.mat')
#         I_x = loadmat('./Data/TrainAzimuth100000/X_imag.mat')
#         self.R_y = self.trans(R_y['Y_real']).squeeze().to(dtype=torch.float32, device=device2)
#         self.I_y = self.trans(I_y['Y_imag']).squeeze().to(dtype=torch.float32, device=device2)
#         self.R_x = self.trans(R_x['X_real']).squeeze().to(dtype=torch.float32, device=device2)
#         self.I_x = self.trans(I_x['X_imag']).squeeze().to(dtype=torch.float32, device=device2)
#         self.length = len(self.R_y)
#
#     def __getitem__(self, item):
#         data = {}
#         data['Y_real'] = self.R_y[item]
#         data['Y_imag'] = self.I_y[item]
#         data['X_real'] = self.R_x[item]
#         data['X_imag'] = self.I_x[item]
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



# class GetData3(Dataset):
#     """
#     return data for dataloader
#     """
#     def __init__(self):
#         R_y = loadmat('./Data/TraintotalAzimuth100000/Y_real.mat')
#         I_y = loadmat('./Data/TraintotalAzimuth100000/Y_imag.mat')
#         R_x = loadmat('./Data/TraintotalAzimuth100000/X_real.mat')
#         I_x = loadmat('./Data/TraintotalAzimuth100000/X_imag.mat')
#         self.R_y = self.trans(R_y['Y_real']).squeeze().to(dtype=torch.float32, device=device3)
#         self.I_y = self.trans(I_y['Y_imag']).squeeze().to(dtype=torch.float32, device=device3)
#         self.R_x = self.trans(R_x['X_real']).squeeze().to(dtype=torch.float32, device=device3)
#         self.I_x = self.trans(I_x['X_imag']).squeeze().to(dtype=torch.float32, device=device3)
#         self.length = len(self.R_y)
#
#     def __getitem__(self, item):
#         data = {}
#         data['Y_real'] = self.R_y[item]
#         data['Y_imag'] = self.I_y[item]
#         data['X_real'] = self.R_x[item]
#         data['X_imag'] = self.I_x[item]
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

# class GetData1(Dataset):
#     """
#     return data for dataloader
#     """
#     def __init__(self):
#         R_y = loadmat('./Data/train1000000ori0to5dB_2to10points/Y_real.mat')
#         I_y = loadmat('./Data/train1000000ori0to5dB_2to10points/Y_imag.mat')
#         R_x = loadmat('./Data/train1000000ori0to5dB_2to10points/X_real.mat')
#         I_x = loadmat('./Data/train1000000ori0to5dB_2to10points/X_imag.mat')
#         self.R_y = self.trans(R_y['Y_real']).squeeze().to(dtype=torch.float32, device=device1)
#         self.I_y = self.trans(I_y['Y_imag']).squeeze().to(dtype=torch.float32, device=device1)
#         self.R_x = self.trans(R_x['X_real']).squeeze().to(dtype=torch.float32, device=device1)
#         self.I_x = self.trans(I_x['X_imag']).squeeze().to(dtype=torch.float32, device=device1)
#         self.length = len(self.R_y)
#
#     def __getitem__(self, item):
#         data = {}
#         data['Y_real'] = self.R_y[item]
#         data['Y_imag'] = self.I_y[item]
#         data['X_real'] = self.R_x[item]
#         data['X_imag'] = self.I_x[item]
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