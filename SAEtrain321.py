import sys
import datetime
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from Model.CVmodel2 import SAE321
from Utils.GetData import GetDatanew
from Utils.GetTestData import GetTestData1
from Utils.Time import Timer
from Utils.Train import TrainfixD32
from Utils.makefile import makefile

def train():
    """config information"""
    cuda0 = "cuda:0" if torch.cuda.is_available() else "cpu"

    ################
    """hyper-parameters"""
    batch_size = 512
    learning_rate = 1e-3
    epoch_num = 1000

    """These configs must be modified every time."""
    paramfilename = f'SAE321 outRD 500000 fixedD32_256 bs{batch_size} {epoch_num}epoch'
    test_name1 = 'Yak42_phaseError'
    Train_flag = False
    ################

    time = Timer()
    device = cuda0
    test_device = "cpu"

    print(f"Using {device} device")
    if Train_flag:
        writer = SummaryWriter(f"TrainLog/{paramfilename}/{datetime.datetime.today().strftime('%Y-%m-%d-%H_%M_%S')}")

    """file configs"""
    path = 'Param/' + paramfilename
    test_path1 = 'Results/' + paramfilename
    test_path = 'Results/' + paramfilename + f'/{test_name1}'
    makefile(path)
    makefile(test_path1)
    makefile(test_path)

    """data"""

    if Train_flag:
        dataset = GetDatanew('TrainRD32_256_500000', device)
        train_dataset, valid_dataset = random_split(
            dataset=dataset,
            lengths=[450000, 50000]
        )

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    """model"""
    model = SAE321(in_features=32, out_features=256)
    model.to(device=device)

    """loss function and optimizer"""
    loss_fn = []
    loss_fn1 = nn.MSELoss()
    loss_fn.append(loss_fn1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)

    """training"""
    Train = TrainfixD32(device)
    if Train_flag:
        time.start()

        for epoch_index in range(epoch_num):
            train_loss = Train.training(train_dataloader, model, loss_fn, optimizer)
            valid_loss = Train.validing(valid_dataloader, model, loss_fn)
            writer.add_scalar('train/loss', train_loss, epoch_index + 1)
            writer.add_scalar('valid/loss', valid_loss, epoch_index + 1)

            if epoch_index % 10 == 0:
                try:
                    for name, parameters in model.named_parameters():
                        writer.add_histogram(f'param/{name}', parameters.detach(), epoch_index + 1)
                        writer.add_histogram(f'grad/{name}', parameters.grad.data, epoch_index + 1)
                except:
                    print(f"{epoch_index}'s histogram has something wrong.")
                    sys.exit(1)

            if epoch_index == 0:
                valid_loss_flag = valid_loss
                param_dict = model.state_dict()
            if valid_loss_flag > valid_loss:
                valid_loss_flag = valid_loss
                flag = epoch_index
                param_dict = model.state_dict()

            if (epoch_index % (epoch_num / 10) == 0) & (epoch_index != 0):
                torch.save(param_dict,
                           f"{path}/bs{batch_size}_"
                           f"lr{str(learning_rate).split('.')[-1]}_"
                           f"eph{epoch_num}"
                           f"_{flag}_minvalidloss.pth")
        print('Training and validating have been finished!')
        print(f'Total training and validating time is: {time.end():.2f} secs')

        """Save model parameters"""
        torch.save(param_dict, f"{path}/bs{batch_size}_"
                               f"lr{str(learning_rate).split('.')[-1]}"
                               f"_eph{epoch_num}"
                               f"_{flag}_totalminvalidloss.pth")
        torch.save(flag, f"{path}/flag.pth")
        print(f'Param of {flag} epoch have been saved!')

    """Test model on measured data."""
    test_dataset = GetTestData1(f'{test_name1}_data32_256')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    flag = torch.load(f"{path}/flag.pth")
    print(flag)
    test_model = SAE321(in_features=32, out_features=256)
    test_model.to(test_device)
    test_model.eval()

    test_model.load_state_dict(torch.load(f"{path}/bs{batch_size}_"
                                          f"lr{str(learning_rate).split('.')[-1]}"
                                          f"_eph{epoch_num}"
                                          f"_{flag}_totalminvalidloss.pth"))


    range_amp = \
    loadmat(f'Data/Test/Dfixed32_256/{test_name1}/{test_name1}_range32_256.mat')[
    f'{test_name1}_range32_256']
    D = loadmat('Data/Test/Dfixed32_256/D32_256.mat')['D']
    Train.testing(test_dataloader, test_model, D, time, range_amp, test_path)



if __name__ == '__main__':
    train()

