"""
将higgs数据转换成libsvm形式
"""

input_file = r"C:\Users\zpengc\Desktop\HIGGS.csv"

train_file = r"C:\Users\zpengc\Desktop\higgs_train.txt"

test_file = r"C:\Users\zpengc\Desktop\higgs_test.txt"

# higgs数据集一共1100万条记录，前1050万条记录作为训练集，后50万条记录作为测试集（higgs数据集网站规定此训练集和测试集的划分）
# https://archive.ics.uci.edu/ml/datasets/HIGGS
num_train = 10500000

# 打开文件
higgs_open = open(input_file, 'r')
train_open = open(train_file, 'w')
test_open = open(test_file, 'w')

line = higgs_open.readline()  # readline()读取一整行，包括最后的'\n'

has_read = 0


# higgs文件29列，第1列是属性(1-signal,0-background)，后面28列是特征值
def write_one_line(tokens, file):
    label = float(tokens[0])
    file.write(str(label))
    for i in range(1, len(tokens)):
        feature_value = float(tokens[i])
        # libsvm数据格式: [label][index1]: [value1][index2]:[value2] …
        file.write(' ' + str(i - 1) + ':' + str(feature_value))
    file.write('\n')


while True:
    tokens = line.split(',')
    if not line:  # 空白行停止读
        break
    if has_read < num_train:
        write_one_line(tokens, train_open)
    else:
        write_one_line(tokens, test_open)
    has_read += 1
    print("has read {} lines from original higgs datafile".format(has_read))
    line = higgs_open.readline()

# 关闭文件
higgs_open.close()
train_open.close()
test_open.close()

