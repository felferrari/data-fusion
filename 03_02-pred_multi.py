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


def pred_model(tm, exp, img_type, test_cond, method):    

    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename='train.log',
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

    print(f'Predicting Experiment {exp} time: {tm}')
    print(f'Conditions: {method}_{img_type}_{test_cond}')

    image_array = np.load(os.path.join(img_path, f'fus_stack_{test_cond}.npy'))
    if img_type == 'OPT':
        image_array = image_array[:, :, :n_opt_layer]
        
    if img_type == 'SAR':
        image_array = image_array[:, :, n_opt_layer:]
        
        
    print('Image stack:', image_array.shape)
    h_, w_, channels = image_array.shape

    path_exp = os.path.join(img_path, 'experiments', f'exp_{exp}')
    path_models = os.path.join(path_exp, 'models')
    path_maps = os.path.join(path_exp, 'pred_maps')

    if not os.path.exists(path_exp):
        os.makedirs(path_exp)   
    if not os.path.exists(path_models):
        os.makedirs(path_models)   
    if not os.path.exists(path_maps):
        os.makedirs(path_maps)

    input_shape = (patch_size, patch_size, channels)
    n_pool = 3
    n_rows = 12#6
    n_cols = 6#3
    rows, cols = image_array.shape[:2]
    pad_rows = rows - np.ceil(rows/(n_rows*2**n_pool))*n_rows*2**n_pool
    pad_cols = cols - np.ceil(cols/(n_cols*2**n_pool))*n_cols*2**n_pool
    print(pad_rows, pad_cols)

    npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
    image1_pad = np.pad(image_array, pad_width=npad, mode='reflect')

    h, w, c = image1_pad.shape
    patch_size_rows = h//n_rows
    patch_size_cols = w//n_cols
    num_patches_x = int(h/patch_size_rows)
    num_patches_y = int(w/patch_size_cols)

    input_shape=(patch_size_rows,patch_size_cols, c)

    if img_type == 'FUS':
        new_model = exp_model(nb_filters, number_class, n_opt_layer)
        new_model.build((None,)+input_shape)
        loss = WBCE(weights = weights)
        optimizers = [
                Adam(lr = 1e-4 , beta_1=0.9),
                Adam(lr = 1e-4 , beta_1=0.9),
                Adam(lr = 1e-4 , beta_1=0.9)
            ]
        new_model.compile(optimizers=optimizers, loss=loss, metrics=['accuracy'])
    else:
        new_model = exp_model(nb_filters, number_class)
        new_model.build((None,)+input_shape)
        adam = Adam(lr = 1e-4 , beta_1=0.9)
        loss = WBCE(weights = weights)
        new_model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        
    new_model.load_weights(os.path.join(path_models, f'{method}_{tm}.h5'))
        
    start_test = time.perf_counter()
    patch_list = []
    patch_list_opt = []
    patch_list_sar = []
    patch_list_fus = []
    
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            patch = image1_pad[patch_size_rows*j:patch_size_rows*(j+1), patch_size_cols*i:patch_size_cols*(i+1), :]
            if img_type == 'FUS':
                pred_opt, pred_sar, pred_fus = new_model.predict(np.expand_dims(patch, axis=0))
                patch_list_opt.append(pred_opt[:,:,:,1])
                patch_list_sar.append(pred_sar[:,:,:,1])
                patch_list_fus.append(pred_fus[:,:,:,1])
                del pred_opt, pred_sar, pred_fus
            else:
                pred = new_model.predict(np.expand_dims(patch, axis=0))
                patch_list.append(pred[:,:,:,1])
                del pred

            del patch 
            
    end_test =  time.perf_counter() - start_test

    if img_type == 'FUS':
        patches_pred_opt = np.asarray(patch_list_opt).astype(np.float32)
        patches_pred_sar = np.asarray(patch_list_sar).astype(np.float32)
        patches_pred_fus = np.asarray(patch_list_fus).astype(np.float32)

        del patches_pred_opt, patches_pred_sar, patches_pred_fus

        prob_recontructed_opt = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred_opt)
        prob_recontructed_sar = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred_sar)
        prob_recontructed_fus = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred_fus)

        del patches_pred_opt, patches_pred_sar, patches_pred_fus

        np.save(os.path.join(path_maps, f'prob_opt_{tm}.npy'),prob_recontructed_opt) 
        np.save(os.path.join(path_maps, f'prob_sar_{tm}.npy'),prob_recontructed_sar) 
        np.save(os.path.join(path_maps, f'prob_fus_{tm}.npy'),prob_recontructed_fus) 
    else:
        patches_pred = np.asarray(patch_list).astype(np.float32)
        del patch_list
        prob_recontructed = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred)
        del patches_pred
        np.save(os.path.join(path_maps, f'prob_{tm}.npy'),prob_recontructed) 
    
    print(f'time prediction model {tm}: {end_test:.2f}')
    np.save(os.path.join(path_exp, f'pred_time_{tm}.npy'), end_test)

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
        for tm in range(times):
            p = Process(target=pred_model, args=(tm, exp,img_types[i], test_cond[i], methods[i]))
            p.start()
            p.join()

            