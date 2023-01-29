import torch
import torch.utils.data
from tqdm import tqdm
from typing import List, Tuple
import numpy as np

from climbing.db import all_climbs, ClimbData, HOLD_ROLE_ID, MAX_Y

class ClimbDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        all_climb_data: List[ClimbData] = [climb.get_climb_data() for climb in tqdm(all_climbs(), "Loading DB")]
        self.climbs: List[Tuple[ClimbData, np.ndarray]] = [(climb_data, climb_data.img()) for climb_data in tqdm(all_climb_data, "Generating images")]
    
    def __getitem__(self, i) -> Tuple[np.ndarray, torch.Tensor]:
        # TODO: trash the transform ToTensor function
        climb_data, img = self.climbs[i]
        img = img.copy()
        img[..., 0] = img[..., 0] / 4
        img[..., 1] = img[..., 1] / 90
        img = self.transform(img).float()
        # -1 because difficulty is {1, 2, ..., 39}
        difficulty_average = torch.tensor(climb_data.difficulty_average - 1)
        return img, difficulty_average

class ClimbDatasetTransformer(torch.utils.data.Dataset):
    MAX_N_HOLDS = 32
    HOLD_DATA_LEN = 2+1

    def __init__(self):
        all_climb_data: List[ClimbData] = [climb.get_climb_data() for climb in tqdm(all_climbs(), "Loading DB")]
        self.climbs = all_climb_data
    
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        climb_data: ClimbData = self.climbs[i]
        difficulty_average = torch.tensor(climb_data.difficulty_average - 1)
        
        # data is one item of a batch input to the model
        # data is concatenation of [angle, -1, -1, ...] and [hold_data] and some padding [-1, -1, -1, ...]
        shape = (ClimbDatasetTransformer.MAX_N_HOLDS, ClimbDatasetTransformer.HOLD_DATA_LEN)
        data = torch.full(shape, fill_value=-1, dtype=torch.float32)

        # insert angle data
        data[0, :] = int(climb_data.angle)

        # insert hold positions and class
        for i, hold in enumerate(climb_data.holds):
            data[i, 0] = hold.x
            data[i, 1] = hold.y
            data[i, 2] = HOLD_ROLE_ID[hold.role]
        
        return data, difficulty_average
        
    def __len__(self):
        return len(self.climbs)