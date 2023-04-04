import glob
import os
import pickle
import re
from river import metrics

def load_model(path):
    model_path = min(glob.iglob(f'{path}/*.pickle'), key=os.path.getctime)
    model=None

    model_timestamp = re.search(r"model_([0-9]+)\.pickle", model_path).group(1)
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


def test(dataset, N:int, subpath:str):    

    model_path=f'models/{subpath}'
    (model, timestamp) = load_model(model_path)

    

    acc= metrics.Accuracy()
    prec= metrics.Precision()
    recl= metrics.Recall()
    f1= metrics.F1()
    i = 0
    try:
        for (X, Y) in dataset:
            i += 0
            if i >= N:
                break;
            Yp = model.predict_one(X)
            
            acc.update(Y, Yp)
            prec.update(Y, Yp)
            recl.update(Y, Yp)
            f1.update(Y, Yp)

            if i % 100 == 0:
                print(f"At {i}")
    except StopIteration as e:
        print(f"Stopped after {i} iterations. Dataset too small.")
    res_path=f'test_results/{subpath}'
    write_results(res_path, timestamp, N, acc, prec, recl, f1)
