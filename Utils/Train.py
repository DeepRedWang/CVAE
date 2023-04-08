from scipy.io import savemat, loadmat
import torch
import numpy as np

class Train():

    def __init__(self, device):
        self.device = device

    def training(self, train_dataloader, model, loss_fn, optimizer):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            a, b = torch.randperm(256)[0:32].sort()
            self.D = torch.eye(256)[a].type(torch.complex64).to(device=self.device)
            Y = data['Y']
            input = torch.einsum('ik,kj->ij', Y, self.D.permute(1,0))
            out = model(input, self.D)
            # loss = loss_fn(R_x_pred, R_x)+loss_fn(I_x_pred, I_x)
            loss = (loss_fn(out.real, Y.real) + loss_fn(out.imag, Y.imag)) / \
                    (loss_fn(Y.real, torch.zeros_like(Y.real)) + loss_fn(Y.imag, torch.zeros_like(Y.imag)) + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = 10 * torch.log(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                a, b = torch.randperm(256)[0:32].sort()
                D = torch.eye(256)[a].type(torch.complex64).to(device=self.device)
                Y = data['Y']
                input = torch.einsum('ik,kj->ij', Y, D.permute(1, 0))
                out = model(input, D)
                loss = loss_fn(out.real, Y.real) + loss_fn(out.imag, Y.imag)
                valid_loss += loss.item()

        valid_loss = 10 * torch.log(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss
    def testing(self, test_dataloader, test_model, D, time, range_amp, test_path):

        for batch_index, data in enumerate(test_dataloader):
            Y = data['test_Y']
            time.start()
            D = torch.tensor(D, dtype=torch.complex64)
            out = test_model(Y, D)
            out = out * torch.tensor(range_amp, dtype=torch.complex64)
            print(f'The forward time is:{time.end():.6f} secs.')
            out = np.array(out.detach())
        savemat(f'{test_path}/' + f'X.mat', {'X': out})

        # max_range = torch.load(f'./Data/test/D/{test_name1}/{test_name1}_range.pth')
        # R_x_pred_ori = out.real.detach() * max_range[f'{test_name1}_range_real']
        # I_x_pred_ori = out.imag.detach() * max_range[f'{test_name1}_range_imag']
        # savemat(f'{test_path}/' + f'X_real.mat', {'X_real': R_x_pred_ori.numpy()})
        # savemat(f'{test_path}/' + f'X_imag.mat', {'X_imag': I_x_pred_ori.numpy()})

class TrainfixD():

    def __init__(self, device):
        self.device = device
        D = loadmat('Data/Test/Dfixed64_256/D64_256.mat')['D']
        self.D = torch.tensor(D, dtype=torch.complex64, device=self.device)

    def training(self, train_dataloader, model, loss_fn, optimizer):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['Y']
            # input = torch.einsum('ik,kj->ij', Y, self.D.permute(1, 0))
            X = data['X']
            out = model(Y, self.D)
            loss = (loss_fn[0](out.real, X.real) + loss_fn[0](out.imag, X.imag)) / \
                    (loss_fn[0](X.real, torch.zeros_like(X.real)) + loss_fn[0](X.imag, torch.zeros_like(X.imag)) + 1e-8)
                   # + 0.05 * loss_fn[1](out, torch.zeros_like(out))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = 10 * torch.log(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['Y']
                X = data['X']
                out = model(Y, self.D)
                loss = loss_fn[0](out.real, X.real) + loss_fn[0](out.imag, X.imag)
                valid_loss += loss.item()

        valid_loss = 10 * torch.log(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss
    def testing(self, test_dataloader, test_model, D, time, range_amp, test_path):

        for batch_index, data in enumerate(test_dataloader):
            Y = data['test_Y']
            D = torch.tensor(D, dtype=torch.complex64)
            time.start()
            out = test_model(Y, D)
            out = out * torch.tensor(range_amp, dtype=torch.complex64)
            print(f'The forward time is:{time.end():.6f} secs.')
            out = np.array(out.detach())
        savemat(f'{test_path}/' + f'X.mat', {'X': out})

class TrainfixD32():

    def __init__(self, device):
        self.device = device
        D = loadmat('Data/Test/Dfixed32_256/D32_256.mat')['D']
        self.D = torch.tensor(D, dtype=torch.complex64, device=self.device)

    def training(self, train_dataloader, model, loss_fn, optimizer):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['Y']
            # input = torch.einsum('ik,kj->ij', Y, self.D.permute(1, 0))
            X = data['X']
            out = model(Y, self.D)

            loss = (loss_fn[0](out.real, X.real) + loss_fn[0](out.imag, X.imag)) / \
                    (loss_fn[0](X.real, torch.zeros_like(X.real)) + loss_fn[0](X.imag, torch.zeros_like(X.imag)) + 1e-8)
            #        # + 0.05 * loss_fn[1](out, torch.zeros_like(out))
            # loss = (loss_fn[2](out.real.softmax(dim=-1).log(), X.real.softmax(dim=-1)) +
            #         loss_fn[2](out.imag.softmax(dim=-1).log(), X.imag.softmax(dim=-1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = 10 * torch.log(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['Y']
                X = data['X']
                out = model(Y, self.D)
                loss = loss_fn[0](out.real, X.real) + loss_fn[0](out.imag, X.imag)
                valid_loss += loss.item()

        valid_loss = 10 * torch.log(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss
    def testing(self, test_dataloader, test_model, D, time, range_amp, test_path):
        for batch_index, data in enumerate(test_dataloader):
            Y = data['test_Y']
            D = torch.tensor(D, dtype=torch.complex64)
            time.start()
            out = test_model(Y, D)
            out = out * torch.tensor(range_amp, dtype=torch.complex64)
            print(f'The forward time is:{time.end():.6f} secs.')
            out = np.array(out.detach())
        savemat(f'{test_path}/' + f'X.mat', {'X': out})




class TrainfixD64():

    def __init__(self, device):
        self.device = device
        D = loadmat('Data/Test/Dfixed64_256/D64_256.mat')['D']
        self.D = torch.tensor(D, dtype=torch.complex64, device=self.device)

    def training(self, train_dataloader, model, loss_fn, optimizer):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['Y']
            # input = torch.einsum('ik,kj->ij', Y, self.D.permute(1, 0))
            X = data['X']
            out = model(Y, self.D)

            loss = (loss_fn[0](out.real, X.real) + loss_fn[0](out.imag, X.imag)) / \
                    (loss_fn[0](X.real, torch.zeros_like(X.real)) + loss_fn[0](X.imag, torch.zeros_like(X.imag)) + 1e-8)
            #        # + 0.05 * loss_fn[1](out, torch.zeros_like(out))
            # loss = (loss_fn[2](out.real.softmax(dim=-1).log(), X.real.softmax(dim=-1)) +
            #         loss_fn[2](out.imag.softmax(dim=-1).log(), X.imag.softmax(dim=-1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = 10 * torch.log(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['Y']
                X = data['X']
                out = model(Y, self.D)
                loss = loss_fn[0](out.real, X.real) + loss_fn[0](out.imag, X.imag)
                valid_loss += loss.item()

        valid_loss = 10 * torch.log(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss
    def testing(self, test_dataloader, test_model, D, time, range_amp, test_path):
        for batch_index, data in enumerate(test_dataloader):
            Y = data['test_Y']
            D = torch.tensor(D, dtype=torch.complex64)
            time.start()
            out = test_model(Y, D)
            out = out * torch.tensor(range_amp, dtype=torch.complex64)
            print(f'The forward time is:{time.end():.6f} secs.')
            out = np.array(out.detach())
        return out