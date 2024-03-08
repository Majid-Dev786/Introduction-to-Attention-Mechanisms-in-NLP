# Importing necessary libraries and modules
import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    # Custom attention layer for sequence-to-sequence models.
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden_state):
        # Calculate attention weights and context vector.
        hidden_state_with_time_axis = tf.expand_dims(hidden_state, 1)
        attention_score = tf.nn.tanh(self.W(features) + self.W(hidden_state_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(attention_score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Encoder(tf.keras.Model):
    # Encoder model for sequence-to-sequence models.
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        # Forward pass through the encoder.
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        # Initializes the hidden state for the encoder.
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    # Decoder model for sequence-to-sequence models.
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = Attention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # Forward pass through the decoder.
        context_vector, attention_weights = self.attention(enc_output, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

# Hyperparameters
vocab_size = 10000
embedding_dim = 256
units = 1024
encoder_units = 1024
decoder_units = 1024
batch_size = 64

# Create Encoder and Decoder instances
encoder = Encoder(vocab_size, embedding_dim, encoder_units, batch_size)
decoder = Decoder(vocab_size, embedding_dim, decoder_units, batch_size)

# Example input and hidden states
example_input = tf.random.uniform((batch_size, 50))
example_hidden = tf.random.uniform((batch_size, encoder_units))
example_enc_output, example_enc_hidden = encoder(example_input, example_hidden)
example_dec_input = tf.random.uniform((batch_size, 1))
example_dec_hidden = example_enc_hidden

# Example predictions and attention weights
example_predictions, example_dec_hidden, example_attention_weights = decoder(example_dec_input,
                                                                             example_dec_hidden,
                                                                             example_enc_output)

# Print shapes of outputs
print('Encoder output shape:', example_enc_output.shape)
print('Decoder output shape:', example_predictions.shape)
print('Attention weights shape:', example_attention_weights.shape)
