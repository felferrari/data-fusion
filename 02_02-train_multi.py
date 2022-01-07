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

def train_model(tm, exp, img_type, train_cond, method):  
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

    print(f'Training Experiment {exp} time: {tm}')
    print(f'Training Conditions: {method}_{img_type}_{train_cond}')

    if train_cond == 'mix':
        image1_array = np.load(os.path.join(img_path, f'fus_stack_n_n.npy'))
        image2_array = np.load(os.path.join(img_path, f'fus_stack_c_c.npy'))

        if img_type == 'OPT':
            image1_array = image1_array[:, :, :n_opt_layer]
            image2_array = image2_array[:, :, :n_opt_layer]
            
        if img_type == 'SAR':
            image1_array = image1_array[:, :, n_opt_layer:]
            image2_array = image2_array[:, :, n_opt_layer:]
            
        print('Image1 stack:', image1_array.shape)
        print('Image2 stack:', image2_array.shape)

        final_mask1 = np.load(os.path.join(img_path, 'final_mask1.npy')).astype(np.uint8)
        print('Labels stack:', final_mask1.shape)
        h_, w_, channels = image1_array.shape

        train_data_loader = PatchesGen(image1_array, final_mask1, patch_size, overlap, grid_size, tiles_tr, 2, batch_size, image2_array)
        val_data_loader = PatchesGen(image1_array, final_mask1, patch_size, overlap, grid_size, tiles_val, 2, batch_size, image2_array)

    
    else:
        image_array = np.load(os.path.join(img_path, f'fus_stack_{train_cond}.npy'))

        if img_type == 'OPT':
            image_array = image_array[:, :, :n_opt_layer]
            
        if img_type == 'SAR':
            image_array = image_array[:, :, n_opt_layer:]
            
        print('Image stack:', image_array.shape)

        final_mask1 = np.load(os.path.join(img_path, 'final_mask1.npy')).astype(np.uint8)
        print('Labels stack:', final_mask1.shape)
        h_, w_, channels = image_array.shape

        train_data_loader = PatchesGen(image_array, final_mask1, patch_size, overlap, grid_size, tiles_tr, 2, batch_size)
        val_data_loader = PatchesGen(image_array, final_mask1, patch_size, overlap, grid_size, tiles_val, 2, batch_size)

    path_exp = os.path.join(img_path, 'experiments', f'exp_{exp}')
    path_models = os.path.join(path_exp, 'models')
    path_maps = os.path.join(path_exp, 'pred_maps')

    if not os.path.exists(path_exp):
        os.makedirs(path_exp)   
    if not os.path.exists(path_models):
        os.makedirs(path_models)   
    if not os.path.exists(path_maps):
        os.makedirs(path_maps)

    datasets = train_data_loader.datasets.astype(np.uint8)
    datasets += 2*val_data_loader.datasets.astype(np.uint8)
    #plt.figure(figsize=(10,5))
    #plt.imshow(datasets, cmap='jet')
    np.save(os.path.join(path_exp,'datasets.npy'), datasets)
    input_shape = (patch_size, patch_size, channels)

    #Training
    epochs=500
    
    
    loss = WBCE(weights = weights)
    

    if img_type == 'FUS':
        
        model = exp_model(nb_filters, number_class, n_opt_layer)
        model.build((None,)+input_shape)
        model.summary()

        #train OPT and SAR for 10 epochs
        lr_schedule_opt = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 5e-4,
            decay_steps=len(train_data_loader),
            decay_rate=0.98,
            staircase=True)
        lr_schedule_sar = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 5e-4,
            decay_steps=len(train_data_loader),
            decay_rate=0.98,
            staircase=True)
        optimizers = [
            Adam(learning_rate = lr_schedule_opt),
            Adam(learning_rate = lr_schedule_sar),
            Adam(learning_rate = 0)
        ]
        model.compile(optimizers=optimizers, loss=loss, metrics=['accuracy'])
        model.set_loss_streams([True, True, False]) #train OPT and SAr networks for 10 epochs
        #earlystop = EarlyStopping(monitor='val_fus_loss', min_delta=0.0001, patience=15, verbose=1, mode='min')
        #checkpoint = ModelCheckpoint(os.path.join(path_models, f'{method}_{tm}.h5'), monitor='val_fus_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        #callbacks_list = [earlystop, checkpoint]
        start_training = time.perf_counter()
        history = model.fit(
            train_data_loader,
            validation_data=val_data_loader,
            epochs=10,
            verbose=2,
            #callbacks=callbacks_list
            )

        lr_schedule_opt = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 5e-4,
            decay_steps=len(train_data_loader),
            decay_rate=0.98,
            staircase=True)
        lr_schedule_sar = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 5e-4,
            decay_steps=len(train_data_loader),
            decay_rate=0.98,
            staircase=True)
        lr_schedule_fus = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 5e-4,
            decay_steps=len(train_data_loader),
            decay_rate=0.98,
            staircase=True)
        optimizers = [
            Adam(learning_rate = lr_schedule_opt),
            Adam(learning_rate = lr_schedule_sar),
            Adam(learning_rate = lr_schedule_fus)
        ]

        model.compile(optimizers=optimizers, loss=loss, metrics=['accuracy'])
        model.set_loss_streams([True, True, True]) #train all networks
        
        earlystop = EarlyStopping(monitor='val_fus_loss', min_delta=0.0001, patience=15, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(os.path.join(path_models, f'{method}_{tm}.h5'), monitor='val_fus_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list = [earlystop, checkpoint]

        history = model.fit(
            train_data_loader,
            validation_data=val_data_loader,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks_list
            )


        
        end_training = time.perf_counter() - start_training
    
    else: #OPT or SAR
        model = exp_model(nb_filters, number_class)
        model.build((None,)+input_shape)
        model.summary()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 5e-4,
            decay_steps=len(train_data_loader),
            decay_rate=0.98,
            staircase=True)

        optimizer = Adam(learning_rate = lr_schedule)
        model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(os.path.join(path_models, f'{method}_{tm}.h5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list = [earlystop, checkpoint]
        start_training = time.perf_counter()
        history = model.fit(
            train_data_loader,
            validation_data=val_data_loader,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks_list)
        end_training = time.perf_counter() - start_training
    
    
    np.save(os.path.join(path_exp, f'time_{tm}.npy'), np.array(end_training))
    np.save(os.path.join(path_exp, f'history_{tm}.npy'), np.array(history.history))

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
            p = Process(target=train_model, args=(tm, exp,img_types[i], train_cond[i], methods[i]))
            p.start()
            p.join()


            