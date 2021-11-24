import sys
import numpy as np
import argparse
import copy
import torch
import torch.nn as nn
import time
from data.sparseloader import DataLoader
from data.data import LibSVMRegData
from models.mlp import MLP_2HL
from models.dynamic_net import DynamicNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam

parser = argparse.ArgumentParser()
parser.add_argument('--feat_d', type=int, default=90)
parser.add_argument('--hidden_d', type=int, default=32)
parser.add_argument('--boost_rate', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--num_nets', type=int, default=40)
parser.add_argument('--data', type=str, default=r"YearPredictionMSD")
parser.add_argument('--tr', type=str, default=r"C:\Users\zpengc\Desktop\music_train.npz")
parser.add_argument('--te', type=str, default=r"C:\Users\zpengc\Desktop\music_test.npz")
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--epochs_per_stage', type=int, default=1)
parser.add_argument('--correct_epoch', type=int, default=1)
parser.add_argument('--L2', type=float, default=0.0e-3)
parser.add_argument('--sparse', action='store_true')  # 默认为false
parser.add_argument('--normalization', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--cv', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--out_f', type=str, default=r"C:\Users\zpengc\Desktop\music_reg.path")
parser.add_argument('--cuda', action='store_true')  # 触发为true，不触发为false，默认为false

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)


def get_data():
    """
    prepare the data and preprocessing
    """
    if opt.data in ['YearPredictionMSD', 'slice_localization']:  # regression数据集
        train_data = LibSVMRegData(opt.tr, opt.feat_d, opt.normalization)
        test_data = LibSVMRegData(opt.te, opt.feat_d, opt.normalization)
        val_data = []
        if opt.cv:  # 增加验证集,95% 5%
            val_data = copy.deepcopy(train_data)
            indices = list(range(len(train_data)))
            cut = int(len(train_data) * 0.95)
            print('Creating Validation set size {}'.format(len(train_data) - cut))
            np.random.shuffle(indices)
            train_idx = indices[:cut]
            val_idx = indices[cut:]

            train_data.feat = train_data.feat[train_idx]
            train_data.label = train_data.label[train_idx]
            val_data.feat = val_data.feat[val_idx]
            val_data.label = val_data.label[val_idx]
    else:
        raise TypeError("该数据集不符合要求，请选择YearPredictionMSD or slice_localization数据集")

    if opt.normalization:  # 对特征的每一列归一化，使均值是0，标准差是1
        scaler = StandardScaler()
        # https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models
        scaler.fit(train_data.feat)
        train_data.feat = scaler.transform(train_data.feat)
        test_data.feat = scaler.transform(test_data.feat)
        if opt.cv:
            val_data.feat = scaler.transform(val_data.feat)
    # 440529+51629+23186 = 515344
    print("train feature shape: {}".format(train_data.feat.shape))  # shape:(440529, 90)
    print("train label shape: {}".format(train_data.label.shape))  # shape:(440529, )
    print("test feature shape: {}".format(test_data.feat.shape))  # shape:(51629, 90)
    print("test label shape: {}".format(test_data.label.shape))  # shape:(51629, )
    print("validation feature shape: {}".format(val_data.feat.shape))  # shape:(23186, 90)
    print("validation label shape: {}".format(val_data.label.shape))  # shape:(23186, 1)
    return train_data, test_data, val_data


# 优化器
def get_optimizer(params, lr, weight_decay):
    opti = Adam(params, lr, weight_decay=weight_decay)
    return opti


def root_mse(net_ensemble, loader):
    loss = 0
    total = 0

    for x, y in loader:
        if opt.cuda:
            x = x.cuda()

        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        # Some operations on tensors cannot be performed on cuda tensors so you need to move them to cpu first.
        # GPU tensor不能直接转为numpy数组，必须先转到CPU tensor
        y = y.cpu().numpy().reshape(len(y), 1)
        out = out.cpu().numpy().reshape(len(y), 1)
        # loss += mean_squared_error(y, out) * len(y)
        loss += mean_squared_error(y, out)
        total += len(y)
    return np.sqrt(loss / total)


if __name__ == "__main__":

    train, test, val = get_data()
    print(opt.data + ' training and test datasets are loaded!')  # 数据集名字

    train_loader = DataLoader(train, opt.batch_size, shuffle=True, num_workers=2)  # 2048 batch size
    test_loader = DataLoader(test, opt.batch_size, shuffle=False, num_workers=2)  # 2048 batch size
    if opt.cv:
        val_loader = DataLoader(val, opt.batch_size, shuffle=True, num_workers=2)  # 2048 batch size

    # best_rmse = pow(10, 6)
    best_rmse = sys.maxsize  # for comparison
    val_rmse = best_rmse
    best_stage = opt.num_nets - 1
    c0 = np.mean(train.label)  # init_gbnn(train)
    net_ensemble = DynamicNet(c0, opt.boost_rate)

    loss_f1 = nn.MSELoss()

    loss_models = torch.zeros((opt.num_nets, 3))  # 存储每次的stage的train_loss, test_loss, val_loss

    stage_time = []

    for stage in range(opt.num_nets):
        print(f"{'='*20}start stage {stage}{'='*20}")
        t0 = time.time()
        model = MLP_2HL.get_model(stage, opt)  # Initialize the model_k: f_k(x), multilayer perception v2
        if opt.cuda:
            model.cuda()

        optimizer = get_optimizer(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train()  # Set the models in ensemble net to train mode
        stage_model_loss = []
        for epoch in range(opt.epochs_per_stage):
            print(f"{'='*20}start epoch {epoch} for stage {stage}{'='*20}")
            # batch size: 2048, 216 batches
            for idx, (x, y) in enumerate(train_loader):  # x shape:(2048, 90)  y shape:(2048, )
                # the last batch shape is (209, 90) (209, )
                if opt.cuda:
                    x = x.cuda()
                    y = torch.as_tensor(y, dtype=torch.float32).cuda().view(-1, 1)  # (2048, 1)  (209, 1)
                else:
                    y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)  # (2048, 1)  (209, 1)
                # penultimate_feat is middle output for concatenating input of new submodel
                penultimate_feat, out = net_ensemble.forward(x)  # penultimate_feat: None, (2048, 32)  out: (), (2048, )
                if opt.cuda:
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)  # (), (2048, 1)
                else:
                    out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)  # (), (2048, 1)
                grad_direction = -(out - y)  # (2048, 1)
                penultimate_out, out = model(x, penultimate_feat)  # penultimate_feat:(2048, 32)  out:(2048, )
                if opt.cuda:
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate * out, grad_direction)  # scalar

                model.zero_grad()
                loss.backward()
                optimizer.step()

                # item(): get a Python number from a tensor containing a single value
                stage_model_loss.append(loss.item() * len(y))
        # complete stage training without fully-corrective step
        net_ensemble.add(model)
        sml = np.sqrt(np.sum(stage_model_loss) / len(train))

        lr_scaler = 3

        # fully-corrective step
        print("start fully corrective step")
        stage_loss = []
        if stage > 0:  # 第一个模型没有修正步骤
            # Adjusting corrective step learning rate
            if stage % 15 == 0:  # 15, 30
                # lr_scaler *= 2
                opt.lr /= 2
                opt.L2 /= 2
            optimizer = get_optimizer(net_ensemble.parameters(), opt.lr / lr_scaler, opt.L2)
            for _ in range(opt.correct_epoch):  # correct_epoch = 1
                stage_loss = []
                for i, (x, y) in enumerate(train_loader):  # 216 batches
                    if opt.cuda:
                        x, y = x.cuda(), y.cuda().view(-1, 1)
                    else:
                        y = y.view(-1, 1)
                    penultimate_result, out = net_ensemble.forward_grad(x)
                    if opt.cuda:
                        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    else:
                        out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                    # https://stackoverflow.com/questions/56856996/difference-in-shape-of-tensor-torch-size-and-torch-size1-in-pytorch
                    loss = loss_f1(out, y)  # 标量/scalar/1-element tensor, shape:()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # item(): get a Python number from a tensor containing a single value
                    stage_loss.append(loss.item() * len(y))  # len(y): 2048, 209
        # store model
        elapsed_tr = time.time() - t0
        stage_time.append(elapsed_tr)
        sl = 0
        if stage_loss:
            sl = np.sqrt(np.sum(stage_loss) / len(train))

        print(f'stage {stage} completed, training time: {elapsed_tr: .1f} sec, model MSE loss: {sml: .5f}, Ensemble Net MSE Loss: {sl: .5f}')

        net_ensemble.to_file(opt.out_f)
        net_ensemble = DynamicNet.from_file(opt.out_f, lambda stage: MLP_2HL.get_model(stage, opt))

        if opt.cuda:
            net_ensemble.to_cuda()
        net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

        # 利用fully-corrective step后的参数重新train、test、valid
        # train
        tr_rmse = root_mse(net_ensemble, train_loader)

        # validation
        if opt.cv:
            val_rmse = root_mse(net_ensemble, val_loader)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_stage = stage

        # test
        te_rmse = root_mse(net_ensemble, test_loader)

        print(f'Stage: {stage}  RMSE@Tr: {tr_rmse:.5f}, RMSE@Val: {val_rmse:.5f}, RMSE@Te: {te_rmse:.5f}')

        loss_models[stage, 0], loss_models[stage, 1], loss_models[stage, 2] = tr_rmse, te_rmse, val_rmse

    net_ensemble.get_summary()
    print(f"{'='*50}")

    # after all stages
    best_train_rmse, best_test_rmse = loss_models[best_stage, 0], loss_models[best_stage, 1]
    print(f'best validation stage: {best_stage}  best_train_rmse: {best_train_rmse:.5f}, best_test_rmse: {best_test_rmse:.5f}')

    loss_models = loss_models.detach().cpu().numpy()
    file_path = './results/' + opt.data + '_rmse'
    print(f"save final model to the path {file_path}")
    np.savez(file_path, rmse=loss_models, params=opt, stage_time=stage_time)

