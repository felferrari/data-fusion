import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=1.0, class_indexes = None, return_sum = False, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.class_indexes = class_indexes
        self.return_sum = return_sum

    def __call__(self, y_true, y_pred): 
        return self.call(y_true, y_pred)
    
    def call(self, y_true, y_pred):
        opt_y_pred = y_pred[0]
        sar_y_pred = y_pred[1]
        fusion_y_pred = y_pred[2]
        alpha = self.alpha
        
        #filter the classes indexes       
        if self.class_indexes is not None:
            y_true = tf.gather(y_true, self.class_indexes, axis=3)
            opt_y_pred = tf.gather(opt_y_pred, self.class_indexes, axis=3)
            sar_y_pred = tf.gather(sar_y_pred, self.class_indexes, axis=3)
            fusion_y_pred = tf.gather(fusion_y_pred, self.class_indexes, axis=3)
            alpha = tf.gather(alpha, self.class_indexes, axis=0)

        opt_y_pred = tf.clip_by_value(opt_y_pred, K.epsilon(), 1.0 - K.epsilon())
        sar_y_pred = tf.clip_by_value(sar_y_pred, K.epsilon(), 1.0 - K.epsilon())
        fusion_y_pred = tf.clip_by_value(fusion_y_pred, K.epsilon(), 1.0 - K.epsilon())

        axes = [0,1,2]

        #opt_loss = - y_true * (alpha * tf.math.pow((1 - opt_y_pred), self.gamma) * tf.math.log(opt_y_pred))
        #sar_loss = - y_true * (alpha * tf.math.pow((1 - sar_y_pred), self.gamma) * tf.math.log(sar_y_pred))
        #fusion_loss = - y_true * (alpha * tf.math.pow((1 - fusion_y_pred), self.gamma) * tf.math.log(fusion_y_pred))

        opt_loss = - y_true * tf.math.pow((1 - opt_y_pred), self.gamma) * tf.math.log(opt_y_pred)
        opt_loss = alpha*tf.math.reduce_mean(opt_loss, axis=axes)
        opt_loss = tf.math.reduce_sum(opt_loss)

        sar_loss = - y_true * tf.math.pow((1 - sar_y_pred), self.gamma) * tf.math.log(sar_y_pred)
        sar_loss = alpha*tf.math.reduce_mean(sar_loss, axis=axes)
        sar_loss = tf.math.reduce_sum(sar_loss)

        fusion_loss = - y_true * tf.math.pow((1 - fusion_y_pred), self.gamma) * tf.math.log(fusion_y_pred)
        fusion_loss = alpha*tf.math.reduce_mean(fusion_loss, axis=axes)
        fusion_loss = tf.math.reduce_sum(fusion_loss)

        if self.return_sum:
            return opt_loss + sar_loss + fusion_loss

        return opt_loss, sar_loss, fusion_loss

class WBCE(Loss):
    def __init__(self, weights=1.0, class_indexes = None, **kwargs):
        super(WBCE, self).__init__(**kwargs)
        self.weights = weights
        self.class_indexes = class_indexes

    def __call__(self, y_true, y_pred, **kwargs): 
        return self.call(y_true, y_pred, **kwargs)
    
    def call(self, y_true, y_pred, **kwargs):
        #opt_y_pred = y_pred
        #sar_y_pred = y_pred[1]
        #fusion_y_pred = y_pred[2]
        weights = self.weights
        
        #filter the classes indexes       
        if self.class_indexes is not None:
            y_true = tf.gather(y_true, self.class_indexes, axis=3)
            y_pred = tf.gather(y_pred, self.class_indexes, axis=3)
            weights = tf.gather(weights, self.class_indexes, axis=0)


        #y_pred /= tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss = y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred)
        loss = loss * weights 
        loss = - tf.math.reduce_mean(loss, -1)
        return tf.math.reduce_mean(loss)

        #if self.return_sum:
        #    return opt_loss #+ sar_loss + fusion_loss

        #return opt_loss#, sar_loss, fusion_loss

class WCCE(Loss):
    def __init__(self, weights=1.0, class_indexes = None, **kwargs):
        super(WCCE, self).__init__(**kwargs)
        self.weights = weights
        self.class_indexes = class_indexes

    def __call__(self, y_true, y_pred, **kwargs): 
        return self.call(y_true, y_pred, **kwargs)
    
    def call(self, y_true, y_pred, **kwargs):
        #opt_y_pred = y_pred
        #sar_y_pred = y_pred[1]
        #fusion_y_pred = y_pred[2]
        weights = self.weights
        
        #filter the classes indexes       
        if self.class_indexes is not None:
            y_true = tf.gather(y_true, self.class_indexes, axis=3)
            y_pred = tf.gather(y_pred, self.class_indexes, axis=3)
            weights = tf.gather(weights, self.class_indexes, axis=0)


        #y_pred /= tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss = y_true * tf.math.log(y_pred)
        loss = loss * weights 
        loss = - tf.math.reduce_mean(loss, -1)
        return tf.math.reduce_mean(loss)

        #if self.return_sum:
        #    return opt_loss #+ sar_loss + fusion_loss

        #return opt_loss#, sar_loss, fusion_loss