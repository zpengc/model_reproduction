import numpy as np
import matplotlib.pyplot as plt


file = np.load("../results/YearPredictionMSD_rmse.npz", allow_pickle=True)

params = file['params']
rmse = file['rmse']
stage_time = file['stage_time']
print(params)
print(stage_time)
print(len(stage_time))

model_train_rmse = []
model_test_rmse = []
model_valid_rmse= []

for idx in range(len(rmse)):
    print(f"read rmse values for model {idx}")
    model_train_rmse.append(rmse[idx][0])
    model_test_rmse.append(rmse[idx][1])
    model_valid_rmse.append(rmse[idx][2])

plt.figure()

plt.plot(range(len(rmse)), model_train_rmse, label="train rmse")
plt.title("train rmse")

plt.plot(range(len(rmse)), model_test_rmse, label="test rmse")
plt.title("test rmse")

plt.plot(range(len(rmse)), model_valid_rmse, label="valid rmse")
plt.title("validation rmse")

plt.legend()
plt.show()


plt.plot(range(len(stage_time)), stage_time, label="stage time")
plt.xlabel("stage iteration")
plt.ylabel("time / second")
plt.title("stage time")
plt.legend()
plt.show()
