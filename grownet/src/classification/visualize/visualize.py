import numpy as np
import matplotlib.pyplot as plt


file = np.load("../results/higgs_auc_score.npy", allow_pickle=True)

print(len(file))

model_train_auc = []
model_test_auc = []
model_valid_auc= []

for idx in range(len(file)):
    print(f"read rmse values for model {idx}")
    model_train_auc.append(file[idx][0])
    model_test_auc.append(file[idx][1])
    model_valid_auc.append(file[idx][2])

plt.figure()

plt.plot(range(len(file)), model_train_auc, label="train auc")
plt.title("train auc")

plt.plot(range(len(file)), model_test_auc, label="test auc")
plt.title("test auc")

plt.plot(range(len(file)), model_valid_auc, label="valid auc")
plt.title("validation auc")

plt.legend()
plt.show()


file = np.load("../results/higgs_cls.npz", allow_pickle=True)
boost_rate = file["dynamic_boostrate"]
plt.plot(range(len(boost_rate)), boost_rate, label="boost_rate")
plt.legend()
plt.show()