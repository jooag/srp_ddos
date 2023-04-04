import numpy as np
from river import utils, metrics
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


def plot_stats(plot_path, idx, timestamp, acc_hist, prec_hist, recl_hist, f1_hist):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)    

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.tight_layout()

    axs[0, 0].plot(idx, acc_hist)
    axs[0, 0].set_title("Accuracy")
    
    axs[0, 1].plot(idx, prec_hist)
    axs[0, 1].set_title("Precision")

    axs[1, 0].plot(idx, recl_hist)
    axs[1, 0].set_title("Recall")

    axs[1, 1].plot(idx, f1_hist)
    axs[1, 1].set_title("F1 Score")    

    fig.savefig(f'{plot_path}/plot_{timestamp}.png', dpi=500)


def train(model, dataset , N:int, window_size:int=100, step_size:int = 100, subpath:str='default' ):    
    
    print("Training initiated. ")
    acc = utils.Rolling(metrics.Accuracy(), window_size)
    prec = utils.Rolling(metrics.Precision(), window_size)
    recl = utils.Rolling(metrics.Recall(), window_size)
    f1 = utils.Rolling(metrics.F1(), window_size)

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
            
            acc.update(Y, Yp)
            prec.update(Y, Yp)
            recl.update(Y, Yp)
            f1.update(Y, Yp)

            model.learn_one(X, Y)

            if i%step_size == 0:
                acc_hist[i//step_size] = acc.get()
                prec_hist[i//step_size] = prec.get()
                recl_hist[i//step_size] = recl.get()
                f1_hist[i//step_size] = f1.get()
                print(f"At {i}")

    except StopIteration as e:
        print(f'Stopped after {i} iterations. Dataset too small.')
    
    timestamp = int((datetime.now() - datetime(1970, 1, 1)).total_seconds())

    model_path = f'models/{subpath}'

    save_model(model, model_path, timestamp)

    idx=np.arange(1, N, step_size)

    plot_path=f'plots/{subpath}'

    plot_stats(plot_path, idx, timestamp, acc_hist, prec_hist, recl_hist, f1_hist)



