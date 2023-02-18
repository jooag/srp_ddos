import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skmultiflow.data import FileStream
from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.meta import StreamingRandomPatchesClassifier
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
import pickle

# pre processamento feito com 
# o codigo de https://github.com/ketlymachado/IoT-IDS-EFDT-DDoS
# uso de 1% dos dados de ddos + todos os dados normais

def test_run():
    start_time = time.time()

    stream = FileStream('shuffled_balanced_dataset.csv')

    #Half Space Trees eh algoritmo de anomaly detection
    #hst = HalfSpaceTrees()

    efdt = ExtremelyFastDecisionTreeClassifier(tie_threshold=0.5, grace_period=100)

    #Quando base_estimator nao eh definido
    #Hoeffding Tree eh o default

    #foram feitos testes para achar o melhor número de estimadores
    srp = StreamingRandomPatchesClassifier(random_state=1,
                                             n_estimators=10, 
                                             base_estimator=efdt)

#####################################
    # Variaveis para controlar loop e performance
    n_samples = 0
    correct_cnt = 0
    correctness_dist = []
    
    max_samples = 200000
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tps = []
    tns = []
    fps = []
    fns = []
    print(f"MAX SAMPLES: {max_samples}")
    # test-then-train loop 
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        y_pred = srp.predict(X)
        if y[0] == y_pred[0]:
            correctness_dist.append(1)
            correct_cnt += 1
            if y[0] == 0:
                tn +=1
            else:
                tp +=1
        else:
            correctness_dist.append(0)
            if y[0] == 0:
                fn +=1
            else:
                fp +=1
        srp.partial_fit(X, y)
        n_samples += 1
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        if n_samples % 100 == 0:
            print(f"TRAIN {n_samples}")

    print("n_samples:{0}".format(str(n_samples)))
    
    with open('model_srp.pickle', 'wb') as f:
        pickle.dump(srp, f)
    ################## MÉTRICAS DO MODELO #####################
    #métricas ao longo da quantidade de samples
    times = [i for i in range(1, n_samples)]
    accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, n_samples)] 
    precisions = [tps[i]/(tps[i] + fps[i]) if (tps[i] + fps[i]) > 0 else 0 for i in range(1, n_samples)]
    recalls = [tps[i]/(tps[i] + fns[i]) if (tps[i] + fns[i]) > 0 else 0 for i in range(1, n_samples)] 
    f1s = [(2*precisions[i]*recalls[i])/(precisions[i]+recalls[i]) if (precisions[i]+recalls[i]) > 0 else 0  for i in range(0, n_samples-1)]

    #métricas finais
    precision = tp/(tp + fp) if tp + fp > 0 else 0
    recall = tp/(tp + fn) if tp + fn > 0 else 0
    f1 = (2*precision*recall)/(precision+recall) if precision+recall > 0 else 0
    

    plt.plot(times, accuracy)
    plt.savefig("srp_acc_eftd.pdf")
    plt.clf()
    plt.plot(times, precisions)
    plt.savefig("srp_p_eftd.pdf")
    plt.clf()
    plt.plot(times, recalls)
    plt.savefig("srp_r_eftd.pdf")
    plt.clf()
    plt.plot(times, f1s)
    plt.savefig("srp_f1_eftd.pdf")
    plt.clf()

    '''
    best_accuracy = np.max(accuracy)
    baccuracy_index = np.argmax(accuracy)

    bprecision = tps[baccuracy_index]/(tps[baccuracy_index] + fps[baccuracy_index]) if tps[baccuracy_index] + fps[baccuracy_index] > 0 else 0
    brecall = tps[baccuracy_index]/(tps[baccuracy_index] + fns[baccuracy_index]) if tps[baccuracy_index] + fns[baccuracy_index] > 0 else 0
    bf1 = (2*bprecision*brecall)/(bprecision+brecall) if bprecision+brecall > 0 else 0
    
    print("baccuracy: {0}".format(str(best_accuracy)))
    print("index: {0}".format(str(baccuracy_index)))
    print("bprecision: {0}".format(str(bprecision)))
    print("brecall: {0}".format(str(brecall)))
    print("bf1: {0}".format(str(bf1)))
    '''

    print("--- Training running time: %.6f seconds ---" % (time.time() - start_time))
    print("last accuracy: {0}".format(str(accuracy[-1])))
    print("last precision: {0}".format(str(precision)))
    print("last recall: {0}".format(str(recall)))
    print("last f1: {0}".format(str(f1)))

    


    ##################### METRICAS COM DADOS NAO VISTOS PELO MODELO ##########
    print("##################### TEST #################################")
    n_samples = 0
    correct_cnt = 0
    correctness_dist = []
    max_samples = 10000
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Somente predicao, nao tem partial fit do modelo
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        y_pred = srp.predict(X)
        if y[0] == y_pred[0]:
            correctness_dist.append(1)
            correct_cnt += 1
            if y[0] == 0:
                tn +=1
            else:
                tp +=1
        else:
            correctness_dist.append(0)
            if y[0] == 0:
                fn +=1
            else:
                fp +=1
        
        n_samples += 1
        if n_samples % 1000 == 0:
            print(f"TEST {n_samples}")

    print("n_samples tested:{0}".format(str(n_samples)))
    
    times = [i for i in range(1, n_samples)]
    acc = sum(correctness_dist)/len(correctness_dist)
    precision = tp/(tp + fp) if tp + fp > 0 else 0
    recall = tp/(tp + fn) if tp + fn > 0 else 0
    f1 = (2*precision*recall)/(precision+recall) if precision+recall > 0 else 0

    print("test accuracy: {0}".format(str(acc)))
    print("test precision: {0}".format(str(precision)))
    print("test recall: {0}".format(str(recall)))
    print("test f1: {0}".format(str(f1)))


if __name__ == "__main__":
    test_run()

