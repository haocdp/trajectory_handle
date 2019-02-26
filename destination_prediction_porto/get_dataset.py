import numpy as np
import matplotlib.pyplot as plt

file_path = "K:/毕业论文/TaxiData_Porto"

destination_new_dataset = np.load(file_path + "/cluster/cluster_dataset.npy")
destination_new_labels = np.load(file_path + "/cluster/destination_labels.npy")

plt.figure()
plt.scatter(destination_new_dataset[:, 0], destination_new_dataset[:, 1], c=destination_new_labels, s=10, cmap='seismic')
plt.title('destination cluster')
plt.show()
