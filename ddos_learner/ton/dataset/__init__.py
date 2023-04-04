from ...common import Dataset
from river import stream

class TONIoTDataset(Dataset):
    
    def __init__(self, train=True):        
        self.path=f"ddos_learner/ton/dataset/processed/{'train' if train else 'test'}_data.csv"
        self.iter = stream.iter_csv(self.path, converters={'duration': float,'src_bytes': float,'dst_bytes':float,'missed_bytes':float,'src_pkts':float,'dst_pkts':float,'label':float}, target='label')

    def __iter__(self):
        return self.iter
