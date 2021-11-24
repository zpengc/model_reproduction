import numpy as np
import sklearn
import argparse
import copy
import time
import torch
import torch.nn as nn
from data.sparseloader import DataLoader
from data.data import LibSVMData
from data.sparse_data import LibSVMDataSp
from models.mlp import MLP_2HL
from models.dynamic_net import DynamicNet
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from misc.auc import auc

parser = argparse.ArgumentParser()
parser.add_argument('--feat_d', type=int, default=28)
parser.add_argument('--hidden_d', type=int, default=16)
parser.add_argument('--boost_rate', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--num_nets', type=int, default=40)
parser.add_argument('--data', type=str, default=r"higgs")
parser.add_argument('--tr', type=str, default=r"C:\Users\zpengc\Desktop\higgs_train.txt")
parser.add_argument('--te', type=str, default=r"C:\Users\zpengc\Desktop\higgs_test.txt")
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--epochs_per_stage', type=int, default=1)
parser.add_argument('--correct_epoch', type=int, default=1)
parser.add_argument('--L2', type=float, default=0.0e-3)
parser.add_argument('--sparse', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--normalization', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--cv', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--model_order', default='second', type=str)
parser.add_argument('--out_f', type=str, default=r"C:\Users\zpengc\Desktop\higgs_cls.path")
parser.add_argument('--cuda', action='store_true')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)  # Sets the number of threads used for intra-operation parallelism on CPU
# torch.get_num_threads() == 8


# prepare the dataset
def get_data():
    """higgs
    This is a classification problem to distinguish between a signal process which produces Higgs bosons
    and a background process which does not.
    1 for signal, 0 for background
    """
    if opt.data == 'higgs':
        # train_data = LibSVMData(opt.tr, opt.feat_d, opt.normalization, 0, 1)
        train_data = LibSVMData(opt.tr, opt.feat_d, opt.normalization, 1, 0)
        # test_data = LibSVMData(opt.te, opt.feat_d, opt.normalization, 0, 1)
        test_data = LibSVMData(opt.te, opt.feat_d, opt.normalization, 1, 0)
    else:
        raise TypeError("该数据集不符合要求，请选择higgs数据集")

    val_data = []
    if opt.cv:
        val_data = copy.deepcopy(train_data)

        # Split the data from cut point
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

    if opt.normalization:
        # https://stackoverflow.com/questions/51237635/difference-between-standard-scaler-and-minmaxscaler
        scaler = MinMaxScaler()  # 压缩到(0,1)之间
        scaler.fit(train_data.feat)
        train_data.feat = scaler.transform(train_data.feat)
        test_data.feat = scaler.transform(test_data.feat)
        if opt.cv:
            val_data.feat = scaler.transform(val_data.feat)

    # 1050 0000 * 95% = 9975000
    print("train feature shape: {}".format(train_data.feat.shape))  # shape:(9975000, 28)
    print("train label shape: {}".format(train_data.label.shape))  # shape:(9975000, )
    print("test feature shape: {}".format(test_data.feat.shape))  # shape:(50 0000, 28)
    print("test label shape: {}".format(test_data.label.shape))  # shape:(50 0000, )
    print("validation feature shape: {}".format(val_data.feat.shape))  # shape:(525000, 28)
    print("validation label shape: {}".format(val_data.label.shape))  # shape:(525000, 1)
    return train_data, test_data, val_data


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    return optimizer


def accuracy(net_ensemble, test_loader):
    correct = 0
    total = 0
    for x, y in test_loader:
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            penultimate_feat, out = net_ensemble.forward(x)
        correct += (torch.sum(y[out > 0.] > 0) + torch.sum(y[out < .0] < 0)).item()  # 正确预测的个数
        total += y.numel()  # 获取tensor元素个数
    return correct / total


def logloss(net_ensemble, loader):
    loss = 0
    total = 0
    loss_f = nn.BCEWithLogitsLoss()  # Binary cross entopy loss with logits, reduction=mean by default
    for x, y in loader:
        if opt.cuda:
            x, y = x.cuda(), y.cuda().view(-1, 1)
        else:
            y = y.view(-1, 1)
        y = (y + 1) / 2
        with torch.no_grad():
            penultimate_feat, out = net_ensemble.forward(x)
        if opt.cuda:
            out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
        else:
            out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
        loss += loss_f(out, y)
        total += 1  # 批次个数

    return loss / total


def auc_score(net_ensemble, loader):
    actual = []
    posterior = []
    for x, y in loader:
        if opt.cuda:
            x = x.cuda()
        with torch.no_grad():
            penultimate_feat, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)  # Why not using the scores themselve than converting to prob
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    score = auc(actual, posterior)
    return score


def init_gbnn(train):
    positive = 0
    negative = 0
    for idx in range(len(train)):
        if train[idx][1] > 0:  # 根据一个特征
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'blind accuracy: {blind_acc}')
    return float(np.log(positive / negative))


if __name__ == "__main__":

    train, test, val = get_data()
    print(opt.data + ' training and test datasets are loaded!')
    train_loader = DataLoader(train, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
    test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
    if opt.cv:
        val_loader = DataLoader(val, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)

    best_score = 0
    val_score = best_score
    best_stage = opt.num_nets - 1

    c0 = init_gbnn(train)  # log of (positive / negative)
    net_ensemble = DynamicNet(c0, opt.boost_rate)  # empty ensemble net without model

    loss_f1 = nn.MSELoss(reduction='none')
    loss_f2 = nn.BCEWithLogitsLoss(reduction='none')

    loss_models = torch.zeros((opt.num_nets, 3))

    all_ensm_losses = []
    all_ensm_losses_te = []
    all_mdl_losses = []
    dynamic_br = []  # boost_ratio
    stage_time = []

    for stage in range(opt.num_nets):  # default 40
        print(f"{'='*20}start stage {stage}{'='*20}")
        t0 = time.time()
        # higgs 100K, 1M , 10M experiment: Subsampling data during training time
        indices = list(range(len(train)))
        split = 1000000  # 1M
        indices = sklearn.utils.shuffle(indices, random_state=41)
        train_idx = indices[:split]  # subsampling after shuffle
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(train, opt.batch_size, sampler=train_sampler, drop_last=True, num_workers=2)

        model = MLP_2HL.get_model(stage, opt)  # Initialize the model_k: f_k(x), multilayer perception v2
        if opt.cuda:
            model.cuda()

        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train()  # Set the models in ensemble net to train mode

        stage_mdlloss = []

        for epoch in range(opt.epochs_per_stage):  # default 1
            for i, (x, y) in enumerate(train_loader):
                if opt.cuda:
                    x, y = x.cuda(), y.cuda().view(-1, 1)
                else:
                    y = y.view(-1, 1)  # x shape:(2048, 28)  y shape:(2048, 1)
                middle_feat, out = net_ensemble.forward(x)  # 前面所有子模型
                if opt.cuda:
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                if opt.model_order == 'first':
                    grad_direction = y / (1.0 + torch.exp(y * out))  # grad_direction shape:(2048, 1)
                else:
                    h = 1 / ((1 + torch.exp(y * out)) * (1 + torch.exp(-y * out)))
                    grad_direction = y * (1.0 + torch.exp(-y * out))
                    out = torch.as_tensor(out)
                    nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                penultimate, out = model(x, middle_feat)  # 新的子模型
                if opt.cuda:
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                else:
                    out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)  # out shape: (2048, 1)

                loss = loss_f1(net_ensemble.boost_rate * out, grad_direction)  # T
                loss = loss * h
                loss = loss.mean()

                model.zero_grad()
                loss.backward()
                optimizer.step()

                stage_mdlloss.append(loss.item())

        net_ensemble.add(model)
        sml = np.mean(stage_mdlloss)

        stage_loss = []
        lr_scaler = 2
        # fully-corrective step
        if stage != 0:
            # Adjusting corrective step learning rate
            if stage % 15 == 0:
                # lr_scaler *= 2
                opt.lr /= 2
                opt.L2 /= 2
            optimizer = get_optim(net_ensemble.parameters(), opt.lr / lr_scaler, opt.L2)
            for _ in range(opt.correct_epoch):
                for i, (x, y) in enumerate(train_loader):
                    if opt.cuda:
                        x, y = x.cuda(), y.cuda().view(-1, 1)
                    else:
                        y = y.view(-1, 1)
                    _, out = net_ensemble.forward_grad(x)
                    if opt.cuda:
                        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    else:
                        out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                    y = (y + 1.0) / 2.0
                    loss = loss_f2(out, y).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_loss.append(loss.item())

        sl_te = logloss(net_ensemble, test_loader)
        # Store dynamic boost rate
        dynamic_br.append(net_ensemble.boost_rate.item())
        # store model
        net_ensemble.to_file(opt.out_f)
        net_ensemble = DynamicNet.from_file(opt.out_f, lambda stage: MLP_2HL.get_model(stage, opt))

        elapsed_tr = time.time() - t0
        sl = 0
        if stage_loss:
            sl = np.mean(stage_loss)

        all_ensm_losses.append(sl)
        all_ensm_losses_te.append(sl_te)
        all_mdl_losses.append(sml)

        stage_time.append(elapsed_tr)
        print(f'stage completed without fully corrective step - {stage}, training time: {elapsed_tr: .1f} sec, boost rate: {net_ensemble.boost_rate: .4f}, Training Loss: {sl: .4f}, Test Loss: {sl_te: .4f}')

        if opt.cuda:
            net_ensemble.to_cuda()
        net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

        # 经过fully-corrective step，参数调整，重新train,test,validation
        print('Acc results from stage := ' + str(stage) + '\n')
        # train
        train_score = auc_score(net_ensemble, train_loader)
        # test
        test_score = auc_score(net_ensemble, test_loader)
        # validation
        if opt.cv:
            val_score = auc_score(net_ensemble, val_loader)
            if val_score > best_score:
                best_score = val_score
                best_stage = stage
        print(f'stage completed with fully corrective step: {stage}, AUC@Val: {val_score:.4f}, AUC@Test: {test_score:.4f}')

        loss_models[stage, 0], loss_models[stage, 1], loss_models[stage, 2] = train_score, test_score, val_score

    # 根据验证集得到的最好结果
    best_train_auc, best_val_auc, best_test_auc = loss_models[best_stage, 0], loss_models[best_stage, 1], loss_models[best_stage, 2]
    print(f'best validation stage: {best_stage},  best_train_auc: {best_train_auc:.4f}, '
          f'best_val_auc: {best_val_auc:.4f}, best_test_auc: {best_test_auc:.4f}')

    # 存储模型
    loss_models = loss_models.detach().cpu().numpy()
    # fname = './results/' + opt.data + '_auc_score'
    # np.save(fname, loss_models)

    # https://www.cnblogs.com/lilu-1226/p/9768368.html
    fname = './results/' + opt.data + '_cls'
    np.savez(fname, training_loss=all_ensm_losses, test_loss=all_ensm_losses_te, model_losses=all_mdl_losses,
             dynamic_boostrate=dynamic_br, params=opt, stage_time=stage_time, loss_models=loss_models)
