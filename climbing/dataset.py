import torch
import torch.utils.data
from tqdm import tqdm
from typing import List, Tuple
import numpy as np

from climbing.db import all_climbs, ClimbData

class ClimbDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        all_climb_data: List[ClimbData] = [climb.get_climb_data() for climb in tqdm(all_climbs(), "Loading DB")]
        self.climbs: List[Tuple[ClimbData, np.ndarray]] = [(climb_data, climb_data.img()) for climb_data in tqdm(all_climb_data, "Generating images")]

    def __getitem__(self, i) -> Tuple[np.ndarray, float]:
        # TODO: trash the transform ToTensor function
        climb_data, img = self.climbs[i]
        img = img.copy()
        img[..., 0] = img[..., 0] / 4
        img[..., 1] = img[..., 1] / 90
        img = self.transform(img).float()
        # -1 because difficulty is {1, 2, ..., 39}
        difficulty_average = torch.tensor(climb_data.difficulty_average - 1)
        return img, difficulty_average
    
    def __len__(self):
        return len(self.climbs)
