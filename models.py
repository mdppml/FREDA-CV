import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.regularizers import Regularizer
# from keras.optimizers import SGD
from keras.optimizers.legacy import SGD


class NMSE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        # Calculating the mse;
        return tf.reduce_mean(tf.square(y_true - y_pred))


class WEN(Regularizer):
    def __init__(self, alpha=0.5, lam=0.5, penalties=None):
        self.alpha = alpha
        self.lam = lam
        self.penalties = penalties

    def __call__(self, x):
        if self.penalties is not None:
            penalties_tensor = tf.reshape(tf.convert_to_tensor(self.penalties, dtype=x.dtype), x.shape)
            l1_penalty = tf.reduce_sum(penalties_tensor * tf.abs(x))
            l2_penalty = tf.reduce_sum(penalties_tensor * tf.square(x))
        else:
            l1_penalty = tf.reduce_sum(tf.abs(x))
            l2_penalty = tf.reduce_sum(tf.square(x))

        regularization = self.alpha * l1_penalty + 0.5 * (1 - self.alpha) * l2_penalty
        return regularization * self.lam

    def get_config(self):
        return {'alpha': float(self.alpha), 'lam': float(self.lam),
                'penalties': self.penalties if self.penalties is not None else None}


def create_model(no_features, alpha, lam, normalized_weights, lr):
    numeric_input = Input(shape=(no_features,))
    output = Dense(1, activation='linear',
                   kernel_regularizer=WEN(alpha=alpha, lam=lam, penalties=normalized_weights),
                   kernel_initializer='zeros')(numeric_input)
    model = Model(inputs=numeric_input, outputs=output)

    optimizer = SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=NMSE(), metrics=['mse'])

    return model


def create_non_adaptive_model(no_features, lr):
    numeric_input = Input(shape=(no_features,))
    output = Dense(1, activation='linear',
                   kernel_initializer='zeros')(numeric_input)
    model = Model(inputs=numeric_input, outputs=output)

    optimizer = SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=NMSE(), metrics=['mse'])

    return model


# Learning rate decay function
def lr_schedule(iteration, total_iterations, initial_lr, final_lr):
    return initial_lr * (final_lr / initial_lr) ** (iteration / total_iterations)
