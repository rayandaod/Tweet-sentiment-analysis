import tensorflow.keras as keras


# Refer to: https://github.com/richliao/textClassifier/issues/28
class AttLayer(keras.layers.Layer):
    def __init__(self, attention_dim):
        self.init = keras.initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = keras.backend.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = keras.backend.variable(self.init((self.attention_dim,)))
        self.u = keras.backend.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = keras.backend.tanh(keras.backend.bias_add(keras.backend.dot(x, self.W), self.b))
        ait = keras.backend.dot(uit, self.u)
        ait = keras.backend.squeeze(ait, -1)

        ait = keras.backend.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= keras.backend.cast(mask, keras.backend.floatx())
        ait /= keras.backend.cast(keras.backend.sum(ait, axis=1, keepdims=True) + keras.backend.epsilon(), keras.backend.floatx())
        ait = keras.backend.expand_dims(ait)
        weighted_input = x * ait
        output = keras.backend.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
