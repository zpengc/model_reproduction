

# 读取LibSVM形式的数据文件
# https://stats.stackexchange.com/questions/61328/libsvm-data-format
def convert2npz(input_filename, out_data_filename):
    input_open = open(input_filename, "r")
    output_feature = open(out_data_filename, "w")
    # output_query = open(out_query_filename,"w")
    # output_label = open(out_query_filename2,"w")

    while True:
        line = input_open.readline()
        if not line:  # 空白行退出
            break
        tokens = line.split(' ')
        tokens[-1] = tokens[-1].strip()
        label = float(tokens[0])
        qid = int(tokens[1].split(':')[1])

        # output_label.write(label + '\n')
        # output_query.write(qid + '\n')
        output_feature.write(str(label) + ' ')
        output_feature.write(str(qid) + ' ')
        output_feature.write(' '.join(tokens[2:]) + '\n')

    input_open.close()
    # output_query.close()
    output_feature.close()
    # output_query2.close()


convert2npz(r"C:\Users\zpengc\Desktop\set1.test.txt", r"C:\Users\zpengc\Desktop\yahoo.train")
convert2npz(r"C:\Users\zpengc\Desktop\set1.test.txt", r"C:\Users\zpengc\Desktop\yahoo.test")
