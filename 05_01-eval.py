from ops import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.dataloader import PatchesGen
from model.losses import WBCE
import time
import tensorflow as tf
import os
import json
import importlib
from multiprocessing import Pool
from multiprocessing import Process
from itertools import repeat
import matplotlib.pyplot as plt
import sys
import logging
import gc


def eval_model(exp, img_type, train_cond, test_cond, method):  

    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename='eval.log',
            filemode='a'
            )
    log = logging.getLogger('foobar')
    sys.stdout = StreamToLogger(log,logging.INFO)
    sys.stderr = StreamToLogger(log,logging.ERROR)

    tf.get_logger().setLevel('ERROR')
    with open(f'experiments.json') as param_file:
        params = json.load(param_file)

    img_path = 'imgs' 
    n_opt_layer = 26 #number of OPT layers, used to split de input data between OPT and SAR

    number_class = 3
    weights = params['weights']
    overlap = params['overlap']
    patch_size = params['patch_size']
    batch_size = params['batch_size']
    nb_filters = params['nb_filters']
    module = importlib.import_module('model.models')
    exp_model = getattr(module, method)
    grid_size = params['grid_size']
    tiles_tr = params['tiles_tr']
    tiles_val = params['tiles_val']

    print(f'Evaluating Experiment {exp}')
    print(f'Conditions: {method}_{img_type}_{train_cond}_{test_cond}')

    final_mask1 = np.lib.format.open_memmap(os.path.join(img_path, 'final_mask1.npy'))
    print('Labels stack:', final_mask1.shape)

    path_exp = os.path.join(img_path, 'experiments', f'exp_{exp}')
    path_models = os.path.join(path_exp, 'models')
    path_maps = os.path.join(path_exp, 'pred_maps')

    if not os.path.exists(path_exp):
        os.makedirs(path_exp)   
    if not os.path.exists(path_models):
        os.makedirs(path_models)   
    if not os.path.exists(path_maps):
        os.makedirs(path_maps)

    datasets = np.load(os.path.join(path_exp,'datasets.npy'))
    test_ds = np.zeros_like(datasets, dtype=np.uint8)
    test_ds[datasets==0] = 1

    mean_prob = np.lib.format.open_memmap(os.path.join(path_maps, 'prob_mean.npy'))

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(121)
    plt.title('Prediction')
    ax1.imshow(mean_prob, cmap ='jet')
    ax1.axis('off')

    ax2 = fig.add_subplot(122)
    plt.title('Reference')
    ax2.imshow(tf.keras.utils.to_categorical(final_mask1, 3)[:,:,1], cmap ='jet')
    ax2.axis('off')
    plt.savefig(os.path.join(path_exp, 'prediction.png'))

    t0 = time.time()
    mean_prob = mean_prob[:final_mask1.shape[0], :final_mask1.shape[1]]

    ref1 = np.ones_like(final_mask1).astype(np.float32)

    ref1 [final_mask1 == 2] = 0
    TileMask = test_ds * ref1
    GTTruePositives = final_mask1==1
        
    Npoints = 50

    Pmax = np.max(mean_prob[GTTruePositives * TileMask ==1])
    ProbList = np.linspace(Pmax,0,Npoints)

    del ref1, TileMask, GTTruePositives
    gc.collect()

    print('Evaluating metrics...')
    metrics = metrics_AP(ProbList, mean_prob, final_mask1, test_ds, 625, 5)
        
    np.save(os.path.join(path_exp, 'acc_metrics.npy'), metrics)
    print(f'elapsed time: {(time.time()-t0)/60} mins')

    metrics_copy = np.array(metrics)
    metrics_copy = complete_nan_values(metrics_copy)

    Recall = metrics_copy[:,0]
    Precision = metrics_copy[:,1]
    AA = metrics_copy[:,2]

    Recall_ = np.insert(Recall, 0, 0)
    Precision_ = np.insert(Precision, 0, Precision[0])
    DeltaR = Recall_[1:]-Recall_[:-1]
    AP = np.sum(Precision_[1:]*DeltaR)
    print('FUSION mAP', AP)

    # Plot Recall vs. Precision curve
    plt.figure(figsize=(10,10))
    plt.plot(metrics_copy[:,0],metrics_copy[:,1], 'b-', label = f'FUSION (AP: {AP:.4f})')
    plt.legend(loc="lower left")
    ax = plt.gca()
    ax.set_ylim([0,1.01])
    ax.set_xlim([0,1.01])
    plt.grid()
    plt.savefig(os.path.join(path_exp, 'result.png'))





if __name__ == '__main__':
    with open(f'experiments.json') as param_file:
        params = json.load(param_file)
    times=params['times']
    exps = []
    img_types = []
    train_cond = []
    test_cond = []
    methods = []
    for exp in params['experiments']:
        exps.append(exp['num'])
        img_types.append(exp['img_type'])
        train_cond.append(exp['train_cond'])
        test_cond.append(exp['test_cond'])
        methods.append(exp['method'])

    for i, exp in enumerate(exps):
        p = Process(target=eval_model, args=(exp,img_types[i], train_cond[i], test_cond[i], methods[i]))
        p.start()
        p.join()

            