import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, n_classes, class_indexes = None, epsilon=1e-5, name=None, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.class_indexes = class_indexes
        self.epsilon = epsilon
        if class_indexes is not None:
            self.shape = len(class_indexes)
        else:
            self.shape = n_classes
        self.true_positives = self.add_weight(name='tp', initializer='zeros', shape = self.shape)
        self.false_positives = self.add_weight(name='fp', initializer='zeros', shape = self.shape)
        self.false_negatives = self.add_weight(name='fn', initializer='zeros', shape = self.shape)

    def update_state(self, y_true, y_pred):
        y_true = tf.clip_by_value(y_true, K.epsilon(), 1.0 - K.epsilon())
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        y_true = tf.one_hot(tf.math.argmax(y_true, axis=-1), self.n_classes)
        y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), self.n_classes)

        if self.class_indexes is not None:
            y_true = tf.gather(y_true, self.class_indexes, axis=3)
            y_pred = tf.gather(y_pred, self.class_indexes, axis=3)

        axes = [1,2]

        tp = tf.math.reduce_sum(y_true * y_pred, axis=axes)
        fp = tf.math.reduce_sum(y_pred, axis=axes) - tp
        fn = tf.math.reduce_sum(y_true, axis=axes) - tp

        self.true_positives.assign_add(tf.math.reduce_sum(tp, axis=0))
        self.false_positives.assign_add(tf.math.reduce_sum(fp, axis=0))
        self.false_negatives.assign_add(tf.math.reduce_sum(fn, axis=0))

    def result(self):
        f1 = (2*self.true_positives + K.epsilon())/(2*self.true_positives + self.false_negatives + self.false_positives + K.epsilon())

        return tf.math.reduce_mean(f1)

    def reset_states(self):
        self.true_positives.assign(tf.zeros(shape = self.shape))
        self.false_positives.assign(tf.zeros(shape = self.shape))
        self.false_negatives.assign(tf.zeros(shape = self.shape))


