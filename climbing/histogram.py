import matplotlib.pyplot as plt
import torch
import climbing
from tqdm import tqdm
import numpy as np

dataset = torch.load("_datasets/cache/climb_dataset_img.pt")

x = np.array([d[1].numpy() for d in tqdm(dataset)])
plt.hist(x, bins=1000)
plt.savefig("histogram.png")
plt.show()