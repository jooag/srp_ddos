from .train import train, load_stats, plot_stats
from .test import test
import os
import pandas as pd

class Dataset():

    #Initializes your dataset
    def __init__(self):    
        pass;       

    def __iter__(self):
        pass;

    def reset(self):
        pass;

def regenerate_plots():
    for subpath in os.listdir('stats'):
        for timestamp in os.listdir(f'stats/{subpath}'):
            
            print(f"{subpath}/{timestamp}")
            df = pd.read_csv(f'stats/{subpath}/{timestamp}')
            timestamp=int(timestamp.replace('.csv', ''))            
            plot_stats(f"plots/{subpath}", **df.to_dict('list'))
