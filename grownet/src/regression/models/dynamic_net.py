from enum import Enum
import torch
import torch.nn as nn


class ForwardType(Enum):
    SIMPLE = 0
    STACKED = 1
    CASCADE = 2
    GRADIENT = 3


class DynamicNet(object):
    """
    ensemble network
    """
    def __init__(self, c0, lr):  # c0表示标签值均值
        self.models = []
        self.c0 = c0
        self.lr = lr
        self.boost_rate = nn.Parameter(torch.tensor(lr, requires_grad=True))

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.append(self.boost_rate)
        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        if len(self.models) == 0:
            return None, self.c0
        penultimate_feat = None  # 倒数第二层输出,和后面模型的输入进行合并,作为新的输入
        prediction = None  # 最后一层输出
        with torch.no_grad():
            for m in self.models:
                if penultimate_feat is None:
                    penultimate_feat, prediction = m(x, penultimate_feat)
                else:
                    penultimate_feat, pred = m(x, penultimate_feat)
                    prediction += pred
        return penultimate_feat, self.c0 + self.boost_rate * prediction

    def forward_grad(self, x):
        if len(self.models) == 0:
            return None, self.c0
        # at least one models
        penultimate_feat = None  # 倒数第二层输出,和后面模型的输入进行合并,作为新的输入
        prediction = None  # 最后一层输出
        for m in self.models:
            if penultimate_feat is None:  # only one model in the ensemble net
                penultimate_feat, prediction = m(x, penultimate_feat)
            else:
                penultimate_feat, pred = m(x, penultimate_feat)
                prediction += pred
        return penultimate_feat, self.c0 + self.boost_rate * prediction

    # 从文件中加载模型
    @classmethod
    def from_file(cls, path, builder):
        d = torch.load(path)
        net = DynamicNet(d['c0'], d['lr'])  # 以字典形式保存模型，就以相应形式读取出来
        net.boost_rate = d['boost_rate']
        for stage, m in enumerate(d['models']):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod)
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {'models': models, 'c0': self.c0, 'lr': self.lr, 'boost_rate': self.boost_rate}
        torch.save(d, path)

    def get_summary(self):
        for model in self.models:
            print(f"{'='*10}model {model}:{'='*10}")
            print(model.parameters())
