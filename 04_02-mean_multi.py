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
import sys
import logging


def mean_model(times, exp, img_type, test_cond, method):   

    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename='mean.log',
            filemode='a'
            )
    log = logging.getLogger('foobar')
    sys.stdout = StreamToLogger(log,logging.INFO)
    sys.stderr = StreamToLogger(log,logging.ERROR)
     
    tf.get_logger().setLevel('ERROR')
    with open(f'experiments_multi.json') as param_file:
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

    print(f'Mean Prediction Experiment {exp}')
    print(f'Conditions: {method}_{img_type}_{test_cond}')

    path_exp = os.path.join(img_path, 'experiments', f'exp_{exp}')
    path_models = os.path.join(path_exp, 'models')
    path_maps = os.path.join(path_exp, 'pred_maps')

    if not os.path.exists(path_exp):
        os.makedirs(path_exp)   
    if not os.path.exists(path_models):
        os.makedirs(path_models)   
    if not os.path.exists(path_maps):
        os.makedirs(path_maps)

    if img_type == 'FUS':
        p0 = np.load(os.path.join(path_maps, f'prob_opt_0.npy')).astype(np.float32)
        prob_rec_opt = np.zeros((p0.shape[0],p0.shape[1], times))
        prob_rec_sar = np.zeros((p0.shape[0],p0.shape[1], times))
        prob_rec_fus = np.zeros((p0.shape[0],p0.shape[1], times))
        del p0

        for tm in range (0, times):
            print(tm)
            prob_rec_opt[:,:,tm] = np.load(os.path.join(path_maps, f'prob_opt_{tm}.npy')).astype(np.float32)
            prob_rec_sar[:,:,tm] = np.load(os.path.join(path_maps, f'prob_sar_{tm}.npy')).astype(np.float32)
            prob_rec_fus[:,:,tm] = np.load(os.path.join(path_maps, f'prob_fus_{tm}.npy')).astype(np.float32)

        mean_prob_opt = np.mean(prob_rec_opt, axis = -1)
        mean_prob_sar = np.mean(prob_rec_sar, axis = -1)
        mean_prob_fus = np.mean(prob_rec_fus, axis = -1)

        np.save(os.path.join(path_maps, f'prob_mean_opt.npy'), mean_prob_opt)
        np.save(os.path.join(path_maps, f'prob_mean_sar.npy'), mean_prob_sar)
        np.save(os.path.join(path_maps, f'prob_mean_fus.npy'), mean_prob_fus)

        for tm in range (0, times):
            os.remove(os.path.join(path_maps, f'prob_opt_{tm}.npy'))
            os.remove(os.path.join(path_maps, f'prob_sar_{tm}.npy'))
            os.remove(os.path.join(path_maps, f'prob_fus_{tm}.npy'))
    else:
        p0 = np.load(os.path.join(path_maps, f'prob_0.npy')).astype(np.float32)
        prob_rec = np.zeros((p0.shape[0],p0.shape[1], times))
        del p0

        for tm in range (0, times):
            print(tm)
            prob_rec[:,:,tm] = np.load(os.path.join(path_maps, f'prob_{tm}.npy')).astype(np.float32)

        mean_prob = np.mean(prob_rec, axis = -1)

        np.save(os.path.join(path_maps, f'prob_mean.npy'), mean_prob)

        for tm in range (0, times):
            os.remove(os.path.join(path_maps, f'prob_{tm}.npy'))

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
        p = Process(target=mean_model, args=(times, exp,img_types[i], test_cond[i], methods[i]))
        p.start()
        p.join()

            