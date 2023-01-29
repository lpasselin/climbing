import matplotlib.pyplot as plt
import torch
import climbing
from tqdm import tqdm
import numpy as np

dataset = torch.load("climb_dataset.pt")

x = np.array([d[1].numpy() for d in tqdm(dataset)])
plt.hist(x, bins=1000)
plt.savefig("histogram.png")
plt.show()