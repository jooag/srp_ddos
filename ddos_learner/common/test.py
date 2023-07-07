import glob
import os
import pickle
import re
from river import metrics

def load_model(path, model_timestamp):
    if model_timestamp is None:
        model_path = max(glob.iglob(f'{path}/*.pickle'), key=os.path.getctime)
        model_timestamp = re.search(r"model_([0-9]+)\.pickle", model_path).group(1)
    else:
        model_path = f"{path}/model_{model_timestamp}.pickle"
    with open(model_path, 'rb') as f:
        model=pickle.load(f)
    return (model, model_timestamp)

def write_results(path, timestamp, N, acc, prec, recl, f1):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/test_res_{timestamp}.txt', 'w') as f:
        f.write('TEST RESULT\n')
        f.write(f'# of samples: {N}\n')
        f.write(f'Accuracy: {acc.get()}\n')
        f.write(f'Precision: {prec.get()}\n')
        f.write(f'Recall: {recl.get()}\n')
        f.write(f'F1: {f1.get()}\n')


def test(dataset, N:int, subpath:str='default', model=None, timestamp=None, save=True):    
    if model is None:
        model_path=f'models/{subpath}'
        (model, timestamp) = load_model(model_path, timestamp)

    dataset.reset()
    acc= metrics.Accuracy()
    prec= metrics.Precision()
    recl= metrics.Recall()
    f1= metrics.F1()
    i = 0   
    done = False
    while not done:
        for (X, Y) in dataset:
            i += 1
            if N > 0 and i >= N:
                done = True
                break;

            Yp = model.predict_one(X)
            
            acc.update(Y, Yp)
            prec.update(Y, Yp)
            recl.update(Y, Yp)
            f1.update(Y, Yp)
            if save:
                if i % 100 == 0:
                    print(f"At {i}")
        
        if not done and N > 0:
            print("Test dataset reached end. Resetting.")
            dataset.reset()
        else:
            done = True
    
    if save:        
        res_path=f'test_results/{subpath}'
        write_results(res_path, timestamp, N, acc, prec, recl, f1)
    return (acc, prec, recl, f1)
