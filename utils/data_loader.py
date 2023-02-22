from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Dataset

data_path = '../data/'
mean = 0.1307
mean_val = 0.1325
std = 0.3081
std_val = 0.3105

trn_datalist = ''
val_datalist = ''

trn_transforms = Compose([
    ToTensor(),
    Normalize(mean, std)
])
val_transforms = Compose([
    ToTensor(),
    Normalize(mean, std)
])

class SampleDataset(Dataset):
    def __init__(self, datalist):
        super().__init__()
        self.datalist = datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        return self.datalist[index]
        
trn_dataset = SampleDataset(data=trn_datalist)
val_dataset = SampleDataset(data=val_datalist)
    

def get_loader(batch_size):
    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader