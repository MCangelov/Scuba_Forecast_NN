from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Layer, TimeDistributed
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam

from keras import Model, backend as K


class TemporalAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(TemporalAttentionLayer, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return output  # Do not sum over the time dimension

    def compute_output_shape(self, input_shape):
        return input_shape  # The output shape is the same as the input shape

    def get_config(self):
        return super(TemporalAttentionLayer, self).get_config()


def create_multiple_LSTM(n_layers: int, units: int, window: int, features: int, dropout: float = 0.0, use_attention: bool = False) -> Model:
    """
    (Optionally) Creates a multi-layer LSTM model with Temporal Attention Mechanism.

    Parameters:
    n_layers (int): The number of LSTM layers.
    units (int): The number of LSTM units.
    window (int): The length of the input sequence.
    features (int): The number of input features.
    dropout (float, optional): The dropout rate. Defaults to 0.0.

    Returns:
    Model: A compiled Keras model.
    """

    model = Sequential()
    for i in range(n_layers):
        model.add(LSTM(units=units, return_sequences=True, activation='relu'))
        if use_attention and i < n_layers - 1:
            model.add(TemporalAttentionLayer())
        model.add(Dropout(dropout))

    model.add(TimeDistributed(Dense(features))) 
    optimizer = Adam(learning_rate=0.0001, clipvalue=0.5)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[
                  RootMeanSquaredError(), MeanAbsoluteError()])
    model.build((None, window, features))

    return model
