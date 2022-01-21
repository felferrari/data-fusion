from ops import *
import time
import tensorflow as tf
import os
import json
import importlib
from multiprocessing import Process
import matplotlib.pyplot as plt
import sys
import logging
import gc



def eval_model(exp, img_type, train_cond, test_cond, method):    

    img_path = 'imgs'

    path_exp = os.path.join(img_path, 'experiments', f'exp_{exp}')
    path_models = os.path.join(path_exp, 'models')
    path_maps = os.path.join(path_exp, 'pred_maps')

    if not os.path.exists(path_exp):
        os.makedirs(path_exp)   
    if not os.path.exists(path_models):
        os.makedirs(path_models)   
    if not os.path.exists(path_maps):
        os.makedirs(path_maps)

    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=os.path.join(path_exp, 'eval.log'),
            filemode='a'
            )
    log = logging.getLogger('eval')
    sys.stdout = StreamToLogger(log,logging.INFO)
    sys.stderr = StreamToLogger(log,logging.ERROR)

    with open(f'experiments_multi.json') as param_file:
        params = json.load(param_file)
 
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

    if img_type == 'FUS':
        mean_prob_opt = np.lib.format.open_memmap(os.path.join(path_maps, 'prob_mean_opt.npy'))
        mean_prob_sar = np.lib.format.open_memmap(os.path.join(path_maps, 'prob_mean_sar.npy'))
        mean_prob_fus = np.lib.format.open_memmap(os.path.join(path_maps, 'prob_mean_fus.npy'))

        fig = plt.figure(figsize=(20,10))
        
        ax1 = fig.add_subplot(141)
        plt.title('Prediction OPT')
        ax1.imshow(mean_prob_opt, cmap ='jet')
        ax1.axis('off')

        ax2 = fig.add_subplot(142)
        plt.title('Prediction SAR')
        ax2.imshow(mean_prob_sar, cmap ='jet')
        ax2.axis('off')

        ax3 = fig.add_subplot(143)
        plt.title('Prediction FUS')
        ax3.imshow(mean_prob_fus, cmap ='jet')
        ax3.axis('off')

        ax4 = fig.add_subplot(144)
        plt.title('Reference')
        ax4.imshow(tf.keras.utils.to_categorical(final_mask1, 3)[:,:,1], cmap ='jet')
        ax4.axis('off')

        plt.savefig(os.path.join(path_exp, 'prediction.png'))

        t0 = time.time()
        mean_prob_opt = mean_prob_opt[:final_mask1.shape[0], :final_mask1.shape[1]]
        mean_prob_sar = mean_prob_sar[:final_mask1.shape[0], :final_mask1.shape[1]]
        mean_prob_fus = mean_prob_fus[:final_mask1.shape[0], :final_mask1.shape[1]]

        ref1 = np.ones_like(final_mask1).astype(np.float32)

        ref1 [final_mask1 == 2] = 0
        TileMask = test_ds * ref1
        GTTruePositives = final_mask1==1
            
        Npoints = 50

        Pmax_opt = np.max(mean_prob_opt[GTTruePositives * TileMask ==1])
        ProbList_opt = np.linspace(Pmax_opt,0,Npoints)

        Pmax_sar = np.max(mean_prob_sar[GTTruePositives * TileMask ==1])
        ProbList_sar = np.linspace(Pmax_sar,0,Npoints)

        Pmax_fus = np.max(mean_prob_fus[GTTruePositives * TileMask ==1])
        ProbList_fus = np.linspace(Pmax_fus,0,Npoints)

        del ref1, TileMask, GTTruePositives
        gc.collect()

        print('Evaluating metrics...')
        metrics_opt = metrics_AP(ProbList_opt, mean_prob_opt, final_mask1, test_ds, 625, 4)
        metrics_sar = metrics_AP(ProbList_sar, mean_prob_sar, final_mask1, test_ds, 625, 4)
        metrics_fus = metrics_AP(ProbList_fus, mean_prob_fus, final_mask1, test_ds, 625, 4)
            
        np.save(os.path.join(path_exp, 'acc_metrics_opt.npy'), metrics_opt)
        np.save(os.path.join(path_exp, 'acc_metrics_sar.npy'), metrics_sar)
        np.save(os.path.join(path_exp, 'acc_metrics_fus.npy'), metrics_fus)
        print(f'elapsed time: {(time.time()-t0)/60} mins')

        metrics_copy_opt = np.array(metrics_opt)
        del metrics_opt
        metrics_copy_opt = complete_nan_values(metrics_copy_opt)

        metrics_copy_sar = np.array(metrics_sar)
        del metrics_sar
        metrics_copy_sar = complete_nan_values(metrics_copy_sar)

        metrics_copy_fus = np.array(metrics_fus)
        del metrics_fus
        metrics_copy_fus = complete_nan_values(metrics_copy_fus)


        Recall_opt = metrics_copy_opt[:,0]
        Precision_opt = metrics_copy_opt[:,1]
        AA_opt = metrics_copy_opt[:,2]

        Recall_sar = metrics_copy_sar[:,0]
        Precision_sar = metrics_copy_sar[:,1]
        AA_sar = metrics_copy_sar[:,2]

        Recall_fus = metrics_copy_fus[:,0]
        Precision_fus = metrics_copy_fus[:,1]
        AA_fus = metrics_copy_fus[:,2]


        Recall_opt = np.insert(Recall_opt, 0, 0)
        Precision_opt = np.insert(Precision_opt, 0, Precision_opt[0])
        DeltaR_opt = Recall_opt[1:]-Recall_opt[:-1]
        AP_opt = np.sum(Precision_opt[1:]*DeltaR_opt)
        print('OPT AP', AP_opt)

        Recall_sar = np.insert(Recall_sar, 0, 0)
        Precision_sar = np.insert(Precision_sar, 0, Precision_sar[0])
        DeltaR_sar = Recall_sar[1:]-Recall_sar[:-1]
        AP_sar = np.sum(Precision_sar[1:]*DeltaR_sar)
        print('SAR AP', AP_sar)

        Recall_fus = np.insert(Recall_fus, 0, 0)
        Precision_fus = np.insert(Precision_fus, 0, Precision_fus[0])
        DeltaR_fus = Recall_fus[1:]-Recall_fus[:-1]
        AP_fus = np.sum(Precision_fus[1:]*DeltaR_fus)
        print('FUSION AP', AP_fus)

        # Plot Recall vs. Precision curve
        plt.figure(figsize=(10,10))
        plt.plot(metrics_copy_opt[:,0],metrics_copy_opt[:,1], 'r-', label = f'OPT (AP: {AP_opt:.4f})')
        plt.plot(metrics_copy_sar[:,0],metrics_copy_sar[:,1], 'g-', label = f'SAR (AP: {AP_sar:.4f})')
        plt.plot(metrics_copy_fus[:,0],metrics_copy_fus[:,1], 'b-', label = f'FUSION (AP: {AP_fus:.4f})')
        plt.legend(loc="lower left")
        ax = plt.gca()
        ax.set_ylim([0,1.01])
        ax.set_xlim([0,1.01])
        plt.grid()
        plt.savefig(os.path.join(path_exp, 'result.png'))
    else:
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
    with open(f'experiments_multi.json') as param_file:
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

            