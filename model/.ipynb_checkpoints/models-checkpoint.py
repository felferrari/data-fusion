import tensorflow as tf
from tensorflow.keras import Model
from .layers import Decoder, Encoder, Classifier, FusionLayer, CombinationLayer, RandomDataAugmentation
from tensorflow.keras.layers import Input
import json
import os
from tensorflow.keras.metrics import BinaryAccuracy
from .metrics import F1Score
from tensorflow.keras.utils import plot_model
import numpy as np 
import math as m
from tqdm import tqdm


#load the params-model.json options
with open(os.path.join('v1', 'params-model.json')) as param_file:
    params_model = json.load(param_file)

def crop_img(img, final_size):
  crop_size = int((img.shape[0] - final_size)/2)
  return img[crop_size:crop_size+final_size, crop_size:crop_size+final_size, :]

class Model_1(Model):
    def __init__(self, **kwargs):
        super(Model_1, self).__init__(**kwargs)
        self.opt_encoder = Encoder(name='opt_encoder')
        self.sar_encoder = Encoder(name='sar_encoder')
        self.decoder = Decoder(name = 'decoder')

        self.fusion = FusionLayer(type='sum')
        
        self.opt_classifier = Classifier(name='opt_classifier')
        self.sar_classifier = Classifier(name='sar_classifier')
        self.fusion_classifier = Classifier(name='fusion_classifier')

        self.combination = CombinationLayer(name = 'combination')
        
        self.accuracy_weights = (1.0/3)*tf.ones(shape=3) #0-OPT, 1-SAR, 2-FUSION
    
    def call(self, inputs, training=True):
        
        opt_enc = self.opt_encoder(inputs[0], training=training)
        sar_enc = self.sar_encoder(inputs[1], training=training)

        x_opt = self.decoder(inputs = opt_enc[0], skip_list = opt_enc[1], training=training)
        x_sar = self.decoder(inputs = sar_enc[0], skip_list = sar_enc[1], training=training)

        x_fusion = self.fusion([x_opt, x_sar])

        y_opt = self.opt_classifier(x_opt, training=training)
        y_sar = self.sar_classifier(x_sar, training=training)
        y_fusion = self.fusion_classifier(x_fusion, training=training)

        return y_opt, y_sar, y_fusion, self.combination((y_opt, y_sar, y_fusion), self.accuracy_weights)

    def summary(self, inputs_shape):
        x = [Input(shape=inputs_shape[0]), Input(shape=inputs_shape[1])]
        model = Model(x, self.call(x))
        return model.summary()

    def plot(self, inputs_shape, to_file='model.png'):
        x = [Input(shape=inputs_shape[0]), Input(shape=inputs_shape[1])]
        model = Model(x, self.call(x))
        plot_model(model, to_file = to_file, show_shapes=True)

    def compile(self, optimizers, loss_fn, metrics_dict, class_weights, class_indexes, **kwargs):
        super(Model_1, self).compile(**kwargs)

        #self.class_indexes = class_indexes

        #initialize optimizers
        self.opt_optimizer = optimizers['opt']
        self.sar_optimizer = optimizers['sar']
        self.fusion_optimizer = optimizers['fusion']

        #set loss function
        #self.loss_fn = loss_fn(alpha = class_weights, class_indexes = class_indexes)
        self.loss_fn = loss_fn(alpha = class_weights, class_indexes = class_indexes)

        #set loss tracker metric
        self.opt_loss_tracker = tf.keras.metrics.Mean(name='opt_loss')
        self.sar_loss_tracker = tf.keras.metrics.Mean(name='sar_loss')
        self.fusion_loss_tracker = tf.keras.metrics.Mean(name='fusion_loss')
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        
        #set accuracy list
        self.accuracy_list = []
        self.accuracy_list.append(BinaryAccuracy(name = 'opt_accuracy'))
        self.accuracy_list.append(BinaryAccuracy(name = 'sar_accuracy'))
        self.accuracy_list.append(BinaryAccuracy(name = 'fusion_accuracy'))
        self.accuracy_list.append(BinaryAccuracy(name = 'combined_accuracy'))

        #set F1Score
        self.f1score_list = []
        self.f1score_list.append(F1Score(name = 'opt_f1score', n_classes=params_model['classes'], class_indexes = class_indexes))
        self.f1score_list.append(F1Score(name = 'sar_f1score', n_classes=params_model['classes'], class_indexes = class_indexes))
        self.f1score_list.append(F1Score(name = 'fusion_f1score', n_classes=params_model['classes'], class_indexes = class_indexes))
        self.f1score_list.append(F1Score(name = 'combined_f1score', n_classes=params_model['classes'], class_indexes = class_indexes))

        #adding other metrics
        self.metrics_list = []
        for m in metrics_dict.keys():
            self.metrics_list.append(metrics_dict[m](name = f'opt_{m}'))
            self.metrics_list.append(metrics_dict[m](name = f'sar_{m}'))
            self.metrics_list.append(metrics_dict[m](name = f'fusion_{m}'))

    def update_losses(self, opt_loss, sar_loss, fusion_loss):
        self.opt_loss_tracker.update_state(opt_loss)
        self.sar_loss_tracker.update_state(sar_loss)
        self.fusion_loss_tracker.update_state(fusion_loss)
        self.loss_tracker.update_state(opt_loss + sar_loss + fusion_loss)

    def get_losses_results(self):
        return {
            'opt_loss': self.opt_loss_tracker.result(),
            'sar_loss': self.sar_loss_tracker.result(),
            'fusion_loss': self.fusion_loss_tracker.result(),
            'loss': self.loss_tracker.result(),
        }

    def update_accuracy(self, y_true, y_pred):
        for i, accuracy in enumerate(self.accuracy_list):
            accuracy.update_state(y_true, y_pred[i])

    def get_accuracy_results(self):
        return {a.name : a.result() for a in self.accuracy_list}

    def update_f1score(self, y_true, y_pred):
        for i, f1score in enumerate(self.f1score_list):
            f1score.update_state(y_true, y_pred[i])

    def get_f1score_results(self):
        return {f.name : f.result() for f in self.f1score_list}

    def update_metrics(self, y_true, y_pred, training):
        for i, m in enumerate(self.metrics_list):
            m.update_state(y_true, y_pred[i%4])

    def get_metrics_results(self):
        return {m.name : m.result() for m in self.metrics_list}

    @property
    def metrics(self):
        return [m for m in self.metrics_list] + [
            self.opt_loss_tracker, 
            self.sar_loss_tracker, 
            self.fusion_loss_tracker, 
            self.loss_tracker] + self.accuracy_list + self.f1score_list
    
 
    def train_step(self, data):
        training = True

        x, y_true = data

        #Data Augmentation
        da_layer = RandomDataAugmentation()
        x, y_true = da_layer(x, y_true)

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.call(x, training=training)
            opt_loss, sar_loss, fusion_loss = self.loss_fn(y_true, y_pred)


            #add regularizers losses to respective loss
            opt_loss += tf.reduce_mean(self.opt_encoder.losses + self.decoder.losses + self.fusion.losses + self.opt_classifier.losses)
            sar_loss += tf.reduce_mean(self.sar_encoder.losses + self.decoder.losses + self.fusion.losses + self.sar_classifier.losses)
            fusion_loss += tf.reduce_mean(self.opt_encoder.losses + self.sar_encoder.losses + self.decoder.losses + self.fusion.losses + self.fusion_classifier.losses)
        
        self.update_losses(opt_loss, sar_loss, fusion_loss)
        self.update_accuracy(y_true, y_pred)
        self.update_f1score(y_true, y_pred)
        self.update_metrics(y_true, y_pred, training = training)
  
        weights = self.trainable_weights

        opt_weights = [w for w in weights if ('opt_' in w.name) or ('decoder' in w.name)]
        sar_weights = [w for w in weights if ('sar_' in w.name) or ('decoder' in w.name)]
        fusion_weights = [w for w in weights if ('opt_encoder' in w.name) or ('sar_encoder' in w.name) or ('decoder' in w.name) or ('fusion_' in w.name)]


        opt_grads = tape.gradient(opt_loss, opt_weights)
        sar_grads = tape.gradient(sar_loss, sar_weights)
        fusion_grads = tape.gradient(fusion_loss, fusion_weights)

        self.opt_optimizer.apply_gradients(zip(opt_grads, opt_weights))
        self.sar_optimizer.apply_gradients(zip(sar_grads, sar_weights))
        self.fusion_optimizer.apply_gradients(zip(fusion_grads, fusion_weights))

        results = self.get_losses_results()
        results.update(self.get_accuracy_results())
        results.update(self.get_f1score_results())
        results.update(self.get_metrics_results())
        return results

    def test_step(self, data):
        training = False
        x, y_true = data

        y_pred = self.call(x, training=training)
        opt_loss, sar_loss, fusion_loss = self.loss_fn(y_true, y_pred)

        self.update_losses(opt_loss, sar_loss, fusion_loss)
        self.update_accuracy(y_true, y_pred)
        self.update_f1score(y_true, y_pred)
        self.update_metrics(y_true, y_pred, training = training)

        results = self.get_losses_results()
        results.update(self.get_accuracy_results())
        results.update(self.get_f1score_results())
        results.update(self.get_metrics_results())
        return results

    def predict_step(self, data):
        training = False
        x = data[0]

        y_pred = self.call(x, training=training)
        
        return y_pred


    def predict_from_patches(self, data, patch_size, patch_stride, batch_size):
        shape = data[0].shape[0:2]
        opt = np.pad(data[0], ((patch_size,patch_size), (patch_size,patch_size), (0,0)), 'symmetric')
        sar = np.pad(data[1], ((patch_size,patch_size), (patch_size,patch_size), (0,0)), 'symmetric')

        #opt_depth = opt.shape[-1]
        #sar_depth = sar.shape[-1]
        depth = 1

        overlap = int((patch_size-patch_stride)/2)

        n_lin = m.ceil(shape[0]/patch_stride)
        n_col = m.ceil(shape[1]/patch_stride)
        n_patches =  n_lin*n_col
        n_batches = m.ceil(n_patches/batch_size)
        opt_reconstructed_img = np.zeros((n_lin*patch_stride, n_col*patch_stride, depth), dtype=np.float16)
        sar_reconstructed_img = np.zeros((n_lin*patch_stride, n_col*patch_stride, depth), dtype=np.float16)
        fusion_reconstructed_img = np.zeros((n_lin*patch_stride, n_col*patch_stride, depth), dtype=np.float16)
        combined_reconstructed_img = np.zeros((n_lin*patch_stride, n_col*patch_stride, depth), dtype=np.float16)

        for batch in tqdm(range(n_batches)):
            patches_n = [p for p in range(batch*batch_size, min(n_patches, (batch+1)*batch_size))]
            opt_patches_l = []
            sar_patches_l = []
            for patch_n in patches_n:
                col = patch_n%n_col
                lin = patch_n//n_col
                l0 = patch_size+lin*patch_stride-overlap
                c0 = patch_size+col*patch_stride-overlap
                opt_patch = opt[l0:l0+patch_size, c0:c0+patch_size, :]
                sar_patch = sar[l0:l0+patch_size, c0:c0+patch_size, :]
                opt_patches_l.append(opt_patch)
                sar_patches_l.append(sar_patch)   
            
            opt_patches_l = np.array(opt_patches_l)
            sar_patches_l = np.array(sar_patches_l)

            opt_pred_b, sar_pred_b, fusion_pred_b, combined_pred_b,  = self.call((opt_patches_l, sar_patches_l))

            
            
            opt_pred_b = opt_pred_b.numpy()
            sar_pred_b = sar_pred_b.numpy()
            fusion_pred_b = fusion_pred_b.numpy()
            combined_pred_b = combined_pred_b.numpy()

            for i, patch_n in enumerate(patches_n):
                col = patch_n%n_col
                lin = patch_n//n_col

                opt_reconstructed_img[lin*patch_stride:lin*patch_stride+patch_stride, col*patch_stride:col*patch_stride+patch_stride] = crop_img(opt_pred_b[i][:,:,1:2], patch_stride)
                sar_reconstructed_img[lin*patch_stride:lin*patch_stride+patch_stride, col*patch_stride:col*patch_stride+patch_stride] = crop_img(sar_pred_b[i][:,:,1:2], patch_stride)
                fusion_reconstructed_img[lin*patch_stride:lin*patch_stride+patch_stride, col*patch_stride:col*patch_stride+patch_stride] = crop_img(fusion_pred_b[i][:,:,1:2], patch_stride)
                combined_reconstructed_img[lin*patch_stride:lin*patch_stride+patch_stride, col*patch_stride:col*patch_stride+patch_stride] = crop_img(combined_pred_b[i][:,:,1:2], patch_stride)

        opt_reconstructed_img = opt_reconstructed_img[:shape[0],:shape[1],:]
        sar_reconstructed_img = sar_reconstructed_img[:shape[0],:shape[1],:]
        fusion_reconstructed_img = fusion_reconstructed_img[:shape[0],:shape[1],:]
        combined_reconstructed_img = combined_reconstructed_img[:shape[0],:shape[1],:]
        
        return opt_reconstructed_img, sar_reconstructed_img, fusion_reconstructed_img, combined_reconstructed_img
                
            

class Model_2(Model):
    def __init__(self, **kwargs):
        super(Model_2, self).__init__(**kwargs)
        self.opt_encoder = Encoder(name='opt_encoder')
        self.sar_encoder = Encoder(name='sar_encoder')
        self.decoder = Decoder(name = 'decoder')

        self.fusion = FusionLayer(type='sum')
        
        self.opt_classifier = Classifier(name='opt_classifier')
        self.sar_classifier = Classifier(name='sar_classifier')
        self.fusion_classifier = Classifier(name='fusion_classifier')

        self.combination = CombinationLayer(name = 'combination')
        
        self.accuracy_weights = (1.0/3)*tf.ones(shape=3) #0-OPT, 1-SAR, 2-FUSION
    
    def call(self, inputs, training=True):
        
        opt_enc = self.opt_encoder(inputs[0], training=training)
        sar_enc = self.sar_encoder(inputs[1], training=training)

        x_opt = self.decoder(inputs = opt_enc[0], skip_list = opt_enc[1], training=training)
        x_sar = self.decoder(inputs = sar_enc[0], skip_list = sar_enc[1], training=training)

        x_fusion = self.fusion([x_opt, x_sar])

        y_opt = self.opt_classifier(x_opt, training=training)
        y_sar = self.sar_classifier(x_sar, training=training)
        y_fusion = self.fusion_classifier(x_fusion, training=training)

        return y_opt, y_sar, y_fusion, self.combination((y_opt, y_sar, y_fusion), self.accuracy_weights)

    def summary(self, inputs_shape):
        x = [Input(shape=inputs_shape[0]), Input(shape=inputs_shape[1])]
        model = Model(x, self.call(x))
        return model.summary()

    def plot(self, inputs_shape, to_file='model.png'):
        x = [Input(shape=inputs_shape[0]), Input(shape=inputs_shape[1])]
        model = Model(x, self.call(x))
        plot_model(model, to_file = to_file, show_shapes=True)

    def compile(self, optimizers, loss_fn, metrics_dict, class_weights, class_indexes, **kwargs):
        super(Model_2, self).compile(**kwargs)

        #self.class_indexes = class_indexes

        #initialize optimizers
        self.opt_optimizer = optimizers['opt']
        self.sar_optimizer = optimizers['sar']
        self.fusion_optimizer = optimizers['fusion']

        #set loss function
        #self.loss_fn = loss_fn(alpha = class_weights, class_indexes = class_indexes)
        self.loss_fn = loss_fn(alpha = class_weights, class_indexes = class_indexes)

        #set loss tracker metric
        self.opt_loss_tracker = tf.keras.metrics.Mean(name='opt_loss')
        self.sar_loss_tracker = tf.keras.metrics.Mean(name='sar_loss')
        self.fusion_loss_tracker = tf.keras.metrics.Mean(name='fusion_loss')
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        
        #set accuracy list
        self.accuracy_list = []
        self.accuracy_list.append(BinaryAccuracy(name = 'opt_accuracy'))
        self.accuracy_list.append(BinaryAccuracy(name = 'sar_accuracy'))
        self.accuracy_list.append(BinaryAccuracy(name = 'fusion_accuracy'))
        self.accuracy_list.append(BinaryAccuracy(name = 'combined_accuracy'))

        #set F1Score
        self.f1score_list = []
        self.f1score_list.append(F1Score(name = 'opt_f1score', n_classes=params_model['classes'], class_indexes = class_indexes))
        self.f1score_list.append(F1Score(name = 'sar_f1score', n_classes=params_model['classes'], class_indexes = class_indexes))
        self.f1score_list.append(F1Score(name = 'fusion_f1score', n_classes=params_model['classes'], class_indexes = class_indexes))
        self.f1score_list.append(F1Score(name = 'combined_f1score', n_classes=params_model['classes'], class_indexes = class_indexes))

        #adding other metrics
        self.metrics_list = []
        for m in metrics_dict.keys():
            self.metrics_list.append(metrics_dict[m](name = f'opt_{m}'))
            self.metrics_list.append(metrics_dict[m](name = f'sar_{m}'))
            self.metrics_list.append(metrics_dict[m](name = f'fusion_{m}'))

    def update_losses(self, opt_loss, sar_loss, fusion_loss):
        self.opt_loss_tracker.update_state(opt_loss)
        self.sar_loss_tracker.update_state(sar_loss)
        self.fusion_loss_tracker.update_state(fusion_loss)
        self.loss_tracker.update_state(opt_loss + sar_loss + fusion_loss)

    def get_losses_results(self):
        return {
            'opt_loss': self.opt_loss_tracker.result(),
            'sar_loss': self.sar_loss_tracker.result(),
            'fusion_loss': self.fusion_loss_tracker.result(),
            'loss': self.loss_tracker.result(),
        }

    def update_accuracy(self, y_true, y_pred):
        for i, accuracy in enumerate(self.accuracy_list):
            accuracy.update_state(y_true, y_pred[i])

    def get_accuracy_results(self):
        return {a.name : a.result() for a in self.accuracy_list}

    def update_f1score(self, y_true, y_pred):
        for i, f1score in enumerate(self.f1score_list):
            f1score.update_state(y_true, y_pred[i])

    def get_f1score_results(self):
        return {f.name : f.result() for f in self.f1score_list}

    def update_metrics(self, y_true, y_pred, training):
        for i, m in enumerate(self.metrics_list):
            m.update_state(y_true, y_pred[i%4])

    def get_metrics_results(self):
        return {m.name : m.result() for m in self.metrics_list}

    @property
    def metrics(self):
        return [m for m in self.metrics_list] + [
            self.opt_loss_tracker, 
            self.sar_loss_tracker, 
            self.fusion_loss_tracker, 
            self.loss_tracker] + self.accuracy_list + self.f1score_list
    
 
    def train_step(self, data):
        training = True

        x, y_true = data

        #Data Augmentation
        da_layer = RandomDataAugmentation()
        x, y_true = da_layer(x, y_true)

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.call(x, training=training)
            opt_loss, sar_loss, fusion_loss = self.loss_fn(y_true, y_pred)


            #add regularizers losses to respective loss
            opt_loss += tf.reduce_sum(self.opt_encoder.losses + self.decoder.losses + self.fusion.losses + self.opt_classifier.losses)
            sar_loss += tf.reduce_sum(self.sar_encoder.losses + self.decoder.losses + self.fusion.losses + self.sar_classifier.losses)
            fusion_loss += tf.reduce_sum(self.opt_encoder.losses + self.sar_encoder.losses + self.decoder.losses + self.fusion.losses + self.fusion_classifier.losses)
        
        self.update_losses(opt_loss, sar_loss, fusion_loss)
        self.update_accuracy(y_true, y_pred)
        self.update_f1score(y_true, y_pred)
        self.update_metrics(y_true, y_pred, training = training)
  
        weights = self.trainable_weights

        opt_weights = [w for w in weights if ('opt_' in w.name) or ('decoder' in w.name)]
        sar_weights = [w for w in weights if ('sar_' in w.name) or ('decoder' in w.name)]
        fusion_weights = [w for w in weights if ('opt_encoder' in w.name) or ('sar_encoder' in w.name) or ('decoder' in w.name) or ('fusion_' in w.name)]


        opt_grads = tape.gradient(opt_loss, opt_weights)
        sar_grads = tape.gradient(sar_loss, sar_weights)
        fusion_grads = tape.gradient(fusion_loss, fusion_weights)

        self.opt_optimizer.apply_gradients(zip(opt_grads, opt_weights))
        self.sar_optimizer.apply_gradients(zip(sar_grads, sar_weights))
        self.fusion_optimizer.apply_gradients(zip(fusion_grads, fusion_weights))

        results = self.get_losses_results()
        results.update(self.get_accuracy_results())
        results.update(self.get_f1score_results())
        results.update(self.get_metrics_results())
        return results

    def test_step(self, data):
        training = False
        x, y_true = data

        y_pred = self.call(x, training=training)
        opt_loss, sar_loss, fusion_loss = self.loss_fn(y_true, y_pred)

        self.update_losses(opt_loss, sar_loss, fusion_loss)
        self.update_accuracy(y_true, y_pred)
        self.update_f1score(y_true, y_pred)
        self.update_metrics(y_true, y_pred, training = training)

        results = self.get_losses_results()
        results.update(self.get_accuracy_results())
        results.update(self.get_f1score_results())
        results.update(self.get_metrics_results())
        return results

    def predict_step(self, data):
        training = False
        x = data[0]

        y_pred = self.call(x, training=training)
        
        return y_pred


    def predict_from_patches(self, data, patch_size, patch_stride, batch_size):
        shape = data[0].shape[0:2]
        opt = np.pad(data[0], ((patch_size,patch_size), (patch_size,patch_size), (0,0)), 'symmetric')
        sar = np.pad(data[1], ((patch_size,patch_size), (patch_size,patch_size), (0,0)), 'symmetric')

        #opt_depth = opt.shape[-1]
        #sar_depth = sar.shape[-1]
        depth = 1

        overlap = int((patch_size-patch_stride)/2)

        n_lin = m.ceil(shape[0]/patch_stride)
        n_col = m.ceil(shape[1]/patch_stride)
        n_patches =  n_lin*n_col
        n_batches = m.ceil(n_patches/batch_size)
        opt_reconstructed_img = np.zeros((n_lin*patch_stride, n_col*patch_stride, depth), dtype=np.float16)
        sar_reconstructed_img = np.zeros((n_lin*patch_stride, n_col*patch_stride, depth), dtype=np.float16)
        fusion_reconstructed_img = np.zeros((n_lin*patch_stride, n_col*patch_stride, depth), dtype=np.float16)
        combined_reconstructed_img = np.zeros((n_lin*patch_stride, n_col*patch_stride, depth), dtype=np.float16)

        for batch in tqdm(range(n_batches)):
            patches_n = [p for p in range(batch*batch_size, min(n_patches, (batch+1)*batch_size))]
            opt_patches_l = []
            sar_patches_l = []
            for patch_n in patches_n:
                col = patch_n%n_col
                lin = patch_n//n_col
                l0 = patch_size+lin*patch_stride-overlap
                c0 = patch_size+col*patch_stride-overlap
                opt_patch = opt[l0:l0+patch_size, c0:c0+patch_size, :]
                sar_patch = sar[l0:l0+patch_size, c0:c0+patch_size, :]
                opt_patches_l.append(opt_patch)
                sar_patches_l.append(sar_patch)   
            
            opt_patches_l = np.array(opt_patches_l)
            sar_patches_l = np.array(sar_patches_l)

            opt_pred_b, sar_pred_b, fusion_pred_b, combined_pred_b,  = self.call((opt_patches_l, sar_patches_l))

            
            
            opt_pred_b = opt_pred_b.numpy()
            sar_pred_b = sar_pred_b.numpy()
            fusion_pred_b = fusion_pred_b.numpy()
            combined_pred_b = combined_pred_b.numpy()

            for i, patch_n in enumerate(patches_n):
                col = patch_n%n_col
                lin = patch_n//n_col

                opt_reconstructed_img[lin*patch_stride:lin*patch_stride+patch_stride, col*patch_stride:col*patch_stride+patch_stride] = crop_img(opt_pred_b[i][:,:,1:2], patch_stride)
                sar_reconstructed_img[lin*patch_stride:lin*patch_stride+patch_stride, col*patch_stride:col*patch_stride+patch_stride] = crop_img(sar_pred_b[i][:,:,1:2], patch_stride)
                fusion_reconstructed_img[lin*patch_stride:lin*patch_stride+patch_stride, col*patch_stride:col*patch_stride+patch_stride] = crop_img(fusion_pred_b[i][:,:,1:2], patch_stride)
                combined_reconstructed_img[lin*patch_stride:lin*patch_stride+patch_stride, col*patch_stride:col*patch_stride+patch_stride] = crop_img(combined_pred_b[i][:,:,1:2], patch_stride)

        opt_reconstructed_img = opt_reconstructed_img[:shape[0],:shape[1],:]
        sar_reconstructed_img = sar_reconstructed_img[:shape[0],:shape[1],:]
        fusion_reconstructed_img = fusion_reconstructed_img[:shape[0],:shape[1],:]
        combined_reconstructed_img = combined_reconstructed_img[:shape[0],:shape[1],:]
        
        return opt_reconstructed_img, sar_reconstructed_img, fusion_reconstructed_img, combined_reconstructed_img
                
             

