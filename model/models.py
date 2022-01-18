import tensorflow as tf
from tensorflow.keras import Model
from .layers import Classifier, FusionLayer
from .layers import UNET_Encoder, UNET_Decoder, CrossFusion, ResUNET_Encoder, ResUNET_Decoder

import json
import os
from tensorflow.keras.metrics import BinaryAccuracy

# load the params-model.json options
with open(os.path.join('v1', 'params-model.json')) as param_file:
    params_model = json.load(param_file)
    
             
class ModelBase(Model):

    def __init__(self, **kwargs):
        super(ModelBase, self).__init__(**kwargs)

    def train_step(self, data):
        training = True
        x = data[0]
        y_true = data[1]

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.call(x, training=training)
            loss = self.compiled_loss(y_true, y_pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.compiled_metrics.update_state(y_true, y_pred)
        self.metrics[0].update_state(loss)
        
        ret_dict = {}
        ret_dict['loss'] = self.metrics[0].result()
        for met in self.compiled_metrics.metrics:
            ret_dict[met.name] = met.result()
        return ret_dict

    def test_step(self, data):
        training = False
        x = data[0]
        y_true = data[1]

        y_pred = self.call(x, training=training)

        loss = self.compiled_loss(y_true, y_pred)

        self.compiled_metrics.update_state(y_true, y_pred)
        self.metrics[0].update_state(loss)
        
        ret_dict = {}
        ret_dict['loss'] = self.metrics[0].result()
        for met in self.compiled_metrics.metrics:
            ret_dict[met.name] = met.result()
        return ret_dict

    def predict_step(self, data):
        training = False
        x = data

        y_pred = self.call(x, training=training)
        return y_pred

class ModelBaseFus(Model):

    def __init__(self, **kwargs):
        super(ModelBaseFus, self).__init__(**kwargs)
        self.recon_weight = 0.0#0.001
        self.loss_streams = [True, True, True]

    def set_loss_streams(self, streams):
        self.loss_streams = streams
    
    def train_step(self, data):
        training = True
        x = data[0]
        y_true = data[1]

        with tf.GradientTape(persistent=True) as tape:
            y_opt, y_sar, y_fus = self.call(x, training=training)


            loss_opt = self.compiled_loss(y_true, y_opt)
            loss_sar = self.compiled_loss(y_true, y_sar)
            loss_fus = self.compiled_loss(y_true, y_fus)

            if hasattr(self, 'fusion'):
                recon_loss = self.recon_weight*tf.math.reduce_sum(self.fusion.recon_losses)
                loss_reg = tf.reduce_sum(self.fusion.losses)
                loss_fus += recon_loss + loss_reg

            loss = loss_opt + loss_sar + loss_fus

        weights = self.trainable_weights

        opt_weights = [w for w in weights if ('opt_' in w.name) or ('shared_' in w.name)]
        sar_weights = [w for w in weights if ('sar_' in w.name) or ('shared_' in w.name)]
        fus_weights = [w for w in weights if ('_encoder' in w.name) or ('_decoder' in w.name) or ('fus_' in w.name)]

        if self.loss_streams[0]:
            opt_grads = tape.gradient(loss_opt, opt_weights)
        if self.loss_streams[1]:
            sar_grads = tape.gradient(loss_sar, sar_weights)
        if self.loss_streams[2]:
            fus_grads = tape.gradient(loss_fus, fus_weights)

        
        if self.loss_streams[0]:
            self.opt_optimizer.apply_gradients(zip(opt_grads, opt_weights))
        if self.loss_streams[1]:
            self.sar_optimizer.apply_gradients(zip(sar_grads, sar_weights))
        if self.loss_streams[2]:
            self.fus_optimizer.apply_gradients(zip(fus_grads, fus_weights))

        self.updateCustomMetrics(
            y_true, y_opt, y_sar, y_fus, loss_opt, loss_sar, loss_fus, loss
            )

        ret_dict = {}
        for met in self.metrics:
            ret_dict[met.name] = met.result()

        return ret_dict

    def test_step(self, data):
        training = False
        x = data[0]
        y_true = data[1]

        y_opt, y_sar, y_fus = self.call(x, training=training)

        loss_opt = self.compiled_loss(y_true, y_opt)
        loss_sar = self.compiled_loss(y_true, y_sar)
        loss_fus = self.compiled_loss(y_true, y_fus)

        if hasattr(self, 'fusion'):
            recon_loss = self.recon_weight*tf.math.reduce_sum(self.fusion.recon_losses)
            loss_reg = tf.reduce_sum(self.fusion.losses)
            loss_fus += recon_loss + loss_reg

        loss = loss_opt + loss_sar + loss_fus

        self.updateCustomMetrics(
            y_true, y_opt, y_sar, y_fus, loss_opt, loss_sar, loss_fus, loss
            )

        ret_dict = {}
        for met in self.metrics:
            ret_dict[met.name] = met.result()

        return ret_dict

    def predict_step(self, data):
        training = False
        x = data

        y_opt, y_sar, y_fus= self.call(x, training=training)


        return y_opt, y_sar, y_fus


    def compile(self, optimizers, metrics, **kwargs):
        super(ModelBaseFus, self).compile(**kwargs)
        '''
        self.opt_optimizer = copy.deepcopy(optimizer)
        self.sar_optimizer = copy.deepcopy(optimizer)
        self.fus_optimizer = copy.deepcopy(optimizer)
        '''
        self.opt_optimizer = optimizers[0]
        self.sar_optimizer = optimizers[1]
        self.fus_optimizer = optimizers[2]

        for metric in metrics:
            if metric == 'accuracy':
                self.opt_accuracy = BinaryAccuracy(name='opt_accuracy')
                self.sar_accuracy = BinaryAccuracy(name='sar_accuracy')
                self.fus_accuracy = BinaryAccuracy(name='fus_accuracy')
        
        #set loss tracker metric
        self.opt_loss_tracker = tf.keras.metrics.Mean(name='opt_loss')
        self.sar_loss_tracker = tf.keras.metrics.Mean(name='sar_loss')
        self.fus_loss_tracker = tf.keras.metrics.Mean(name='fus_loss')
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    #to reset all metrics between epochs
    @property
    def metrics(self):
        return [
            self.opt_accuracy, self.sar_accuracy , self.fus_accuracy,
            self.opt_loss_tracker, self.sar_loss_tracker, self.fus_loss_tracker, self.loss_tracker
        ]
    
    def updateCustomMetrics(self, y_true, y_opt, y_sar, y_fus, loss_opt, loss_sar, loss_fus, loss):
        self.opt_accuracy.update_state(y_true, y_opt)
        self.sar_accuracy.update_state(y_true, y_sar)
        self.fus_accuracy.update_state(y_true, y_fus)

        self.opt_loss_tracker.update_state(loss_opt)
        self.sar_loss_tracker.update_state(loss_sar)
        self.fus_loss_tracker.update_state(loss_fus)
        self.loss_tracker.update_state(loss)       

class UNET(ModelBase):
    def __init__(self, filters, n_classes, **kwargs):
        super(UNET, self).__init__(**kwargs)
        self.encoder = UNET_Encoder(filters, name = 'unet_encoder')
        self.decoder = UNET_Decoder(filters, n_classes, name = 'unet_decoder')
        self.classifier = Classifier(n_classes, name='classifier')

    def call(self, inputs, training = False):
        x = inputs
        x = self.encoder(x, training = training)
        x = self.decoder(x, training = training)
        x = self.classifier(x, training = training)

        return x

class ResUNET(ModelBase):
    def __init__(self, filters, n_classes, **kwargs):
        super(ResUNET, self).__init__(**kwargs)
        self.encoder = ResUNET_Encoder(filters, name = 'resunet_encoder')
        self.decoder = ResUNET_Decoder(filters, name = 'resunet_decoder')
        self.classifier = Classifier(n_classes, name='classifier')

    def call(self, inputs, training = False):
        x = self.encoder(inputs, training = training)
        x = self.decoder(x, training = training)
        x = self.classifier(x, training = training)

        return x

class CrossFusion_UNET(ModelBaseFus):
    def __init__(self, filters, n_classes, n_opt_layers, **kwargs):
        super(CrossFusion_UNET, self).__init__(**kwargs)
        self.n_opt_layers = n_opt_layers

        self.opt_encoder = UNET_Encoder(filters, name = 'opt_encoder')
        self.sar_encoder = UNET_Encoder(filters, name = 'sar_encoder')

        self.opt_decoder = UNET_Decoder(filters, n_classes, name = 'opt_decoder')
        self.sar_decoder = UNET_Decoder(filters, n_classes, name = 'sar_decoder')
        
        self.fusion = CrossFusion(params_model['fusion']['filters'], name='fus_cross')

        self.opt_classifier = Classifier(n_classes, name='opt_classifier')
        self.sar_classifier = Classifier(n_classes, name='sar_classifier')
        #self.fus_classifier = Classifier(name='fus_classifier')


    def call(self, inputs, training = False):
        x_opt = inputs[:,:,:,:self.n_opt_layers]
        x_sar = inputs[:,:,:,self.n_opt_layers:]

        opt_enc = self.opt_encoder(x_opt, training = training)
        sar_enc = self.sar_encoder(x_sar, training = training)

        opt_dec = self.opt_decoder(opt_enc, training = training)
        sar_dec = self.sar_decoder(sar_enc, training = training)

        fus_out = self.fusion([opt_dec, sar_dec], training=training)

        opt_out = self.opt_classifier(opt_dec, training = training)
        sar_out = self.sar_classifier(sar_dec, training = training)


        return opt_out, sar_out, fus_out

class CrossFusion_ResUNET(ModelBaseFus):
    def __init__(self, filters, n_classes, n_opt_layers, **kwargs):
        super(CrossFusion_ResUNET, self).__init__(**kwargs)
        self.n_opt_layers = n_opt_layers

        self.opt_encoder = ResUNET_Encoder(filters, name = 'opt_encoder')
        self.sar_encoder = ResUNET_Encoder(filters, name = 'sar_encoder')

        self.opt_decoder = ResUNET_Decoder(filters, name = 'opt_decoder')
        self.sar_decoder = ResUNET_Decoder(filters, name = 'sar_decoder')
        
        self.fusion = CrossFusion(params_model['fusion']['filters'], name='fus_cross')

        self.opt_classifier = Classifier(n_classes, name='opt_classifier')
        self.sar_classifier = Classifier(n_classes, name='sar_classifier')


    def call(self, inputs, training = False):
        x_opt = inputs[:,:,:,:self.n_opt_layers]
        x_sar = inputs[:,:,:,self.n_opt_layers:]

        opt_enc = self.opt_encoder(x_opt, training = training)
        sar_enc = self.sar_encoder(x_sar, training = training)

        opt_dec = self.opt_decoder(opt_enc, training = training)
        sar_dec = self.sar_decoder(sar_enc, training = training)

        fus_out = self.fusion([opt_dec, sar_dec], training=training)

        opt_out = self.opt_classifier(opt_dec, training = training)
        sar_out = self.sar_classifier(sar_dec, training = training)


        return opt_out, sar_out, fus_out

class Concat_UNET(ModelBaseFus):
    def __init__(self, filters, n_classes, n_opt_layers, **kwargs):
        super(Concat_UNET, self).__init__(**kwargs)
        self.n_opt_layers = n_opt_layers

        self.opt_encoder = UNET_Encoder(filters, name = 'opt_encoder')
        self.sar_encoder = UNET_Encoder(filters, name = 'sar_encoder')

        self.opt_decoder = UNET_Decoder(filters, n_classes, name = 'opt_decoder')
        self.sar_decoder = UNET_Decoder(filters, n_classes, name = 'sar_decoder')
        
        self.fusion = FusionLayer(type='concat', name='fus_concat')

        self.opt_classifier = Classifier(n_classes, name='opt_classifier')
        self.sar_classifier = Classifier(n_classes, name='sar_classifier')
        self.fus_classifier = Classifier(n_classes, name='fus_classifier')



    def call(self, inputs, training = False):
        x_opt = inputs[:,:,:,:self.n_opt_layers]
        x_sar = inputs[:,:,:,self.n_opt_layers:]

        opt_enc = self.opt_encoder(x_opt, training = training)
        sar_enc = self.sar_encoder(x_sar, training = training)

        opt_dec = self.opt_decoder(opt_enc, training = training)
        sar_dec = self.sar_decoder(sar_enc, training = training)

        fus_out = self.fusion([opt_dec, sar_dec], training=training)

        opt_out = self.opt_classifier(opt_dec, training = training)
        sar_out = self.sar_classifier(sar_dec, training = training)
        fus_out = self.fus_classifier(fus_out, training = training)


        return opt_out, sar_out, fus_out

class Concat_ResUNET(ModelBaseFus):
    def __init__(self, filters, n_classes, n_opt_layers, **kwargs):
        super(Concat_ResUNET, self).__init__(**kwargs)
        self.n_opt_layers = n_opt_layers

        self.opt_encoder = ResUNET_Encoder(filters, name = 'opt_encoder')
        self.sar_encoder = ResUNET_Encoder(filters, name = 'sar_encoder')

        self.opt_decoder = ResUNET_Decoder(filters, name = 'opt_decoder')
        self.sar_decoder = ResUNET_Decoder(filters, name = 'sar_decoder')
        
        self.fusion = FusionLayer(type='concat', name='fus_concat')

        self.opt_classifier = Classifier(n_classes, name='opt_classifier')
        self.sar_classifier = Classifier(n_classes, name='sar_classifier')
        self.fus_classifier = Classifier(n_classes, name='fus_classifier')



    def call(self, inputs, training = False):
        x_opt = inputs[:,:,:,:self.n_opt_layers]
        x_sar = inputs[:,:,:,self.n_opt_layers:]

        opt_enc = self.opt_encoder(x_opt, training = training)
        sar_enc = self.sar_encoder(x_sar, training = training)

        opt_dec = self.opt_decoder(opt_enc, training = training)
        sar_dec = self.sar_decoder(sar_enc, training = training)

        fus_out = self.fusion([opt_dec, sar_dec], training=training)

        opt_out = self.opt_classifier(opt_dec, training = training)
        sar_out = self.sar_classifier(sar_dec, training = training)
        fus_out = self.fus_classifier(fus_out, training = training)



        return opt_out, sar_out, fus_out