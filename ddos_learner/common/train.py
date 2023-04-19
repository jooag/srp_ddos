import numpy as np
from river import utils, metrics
from .test import test
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

def save_model(model, path, timestamp):

    

    if not os.path.exists(path):
        print(path)
        os.makedirs(path)

    with open(f'{path}/model_{timestamp}.pickle', 'wb') as f:
        pickle.dump(model, f)

def save_stats(path, timestamp, acc, prec, recl, f1):

    if not os.path.exists(path):        
        os.makedirs(path)

    with open(f'{path}/model_{timestamp}.pickle', 'wb') as f:
        pickle.dump({'acc':acc, 'prec':prec, 'recl':recl, 'f1':f1}, f)


def plot_stats(plot_path, idx, timestamp, acc_hist, prec_hist, recl_hist, f1_hist):
    plot_path=f'{plot_path}/{timestamp}'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)    
    
    for (name, data) in zip(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [acc_hist, prec_hist, recl_hist, f1_hist]):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.tight_layout()

        ax.plot(idx, data)
        ax.set_ylim(0, 1)
        ax.set_title(name)
        
        fig.savefig(f'{plot_path}/{name}.png', dpi=500)



def train(model, dataset , test_dataset, N:int, step_size:int = 100, subpath:str='default' ):    
    
    print("Training initiated. ")

    it = iter(dataset)

    acc_hist = np.empty(N//step_size)
    prec_hist = np.empty(N//step_size)
    recl_hist = np.empty(N//step_size)
    f1_hist = np.empty(N//step_size)
    i = 0

    try:
        for (X, Y) in dataset:
            i += 1
            if i >= N:
                break;
            Yp = model.predict_one(X)     

            model.learn_one(X, Y)

            if i%step_size == 0:
                print(f"At {i}.")
                (acc, prec, recl, f1) = test(test_dataset, 0.05*N, model=model, save=False)

                acc_hist[i//step_size] = acc.get()
                prec_hist[i//step_size] = prec.get()
                recl_hist[i//step_size] = recl.get()
                f1_hist[i//step_size] = f1.get()
                print(f"Acc: {acc.get():.8f}. Prec: {prec.get():.8f}.\nRecl: {recl.get():.8f}. F1: {f1.get():.8f}")

    except StopIteration as e:
        print(f'Stopped after {i} iterations. Dataset too small.')
    
    timestamp = int((datetime.now() - datetime(1970, 1, 1)).total_seconds())

    model_path = f'models/{subpath}'

    save_model(model, model_path, timestamp)

    stats_path=f'stats/{subpath}'

    save_stats(stats_path, timestamp, acc_hist, prec_hist, recl_hist, f1_hist)

    idx=np.arange(1, N, step_size)

    plot_path=f'plots/{subpath}'

    plot_stats(plot_path, idx, timestamp, acc_hist, prec_hist, recl_hist, f1_hist)



