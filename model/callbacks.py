# -*- coding: utf-8 -*-

from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class UpdateWeights(Callback):
   
    def on_epoch_end(self, batch, logs = None):
        self.model.combine_weights.updateWeights(logs)

