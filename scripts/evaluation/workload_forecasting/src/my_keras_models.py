from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, InputLayer, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D

def reset_model_states(model):
	for layer in model.layers:
		if hasattr(layer, 'reset_states'):
			layer.reset_states()

def get_lstm_model(shape, neurons, stateful=True, class_count=1, dense_hidden_layers=[], loss="mse", optimizer="adam", activation="linear"):
	model = Sequential()
	model.add(InputLayer(batch_input_shape=shape))
	model.add(LSTM(neurons, stateful=stateful))
	for neurons in dense_hidden_layers:
		model.add(Dense(neurons))
	model.add(Dense(class_count, activation=activation))
	model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
	print(model.summary())
	return model

def get_mlp_model(shape, in_neurons, class_count, dense_layers, loss, optimizer, activation):
	# model = Sequential()
	input_layer = Input(shape=shape)
	hidden_layers = [Dense(in_neurons) (input_layer)]
	output_layer = list()
	# model.add(Dense(in_neurons, input_shape=shape))
	for neurons in dense_layers:
		# model.add(Dense(neurons))
		hidden_layers.append(Dense(neurons) (hidden_layers[-1]))
	# model.add(Dense(class_count))
	if isinstance(class_count, int):
		output_layer = [Dense(class_count) (hidden_layers[-1])]
	elif len(class_count) == 1:
		output_layer = [Dense(class_count[0]) (hidden_layers[-1])]
	else:
		for out_neurons in class_count:
			output_layer.append(Dense(out_neurons) (hidden_layers[-1]))
	# model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
	model = Model(inputs=[input_layer], outputs=output_layer)
	model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
	print(model.summary())
	return model

def get_stacked_lstm_model(shape, neurons, stateful=True, class_count=1, loss="mse", optimizer="adam", stacked_layers=2, activation="linear"):
	model = Sequential()
	# Input LSTM layer
	model.add(InputLayer(batch_input_shape=shape))
	model.add(LSTM(neurons, stateful=stateful, return_sequences=True))
	# Stacked LSTM hidden layers
	for _ in range(2, stacked_layers):
		model.add(LSTM(neurons, stateful=stateful, return_sequences=True))
	# Last LSTM hidden layer
	model.add(LSTM(neurons, stateful=stateful))
	model.add(Dense(class_count, activation=activation))
	model.compile(loss=loss, optimizer=optimizer)
	print(model.summary())
	return model

def get_transformer_model(shape, neurons, num_heads=4, class_count=1, loss="mse", optimizer="adam", activation="linear", dropout_rate=0.1):
	"""
	Transformer model for time series forecasting.
	shape: (batch_size, timesteps, features) for transformer
	neurons: dimension for the transformer embedding/feed-forward network
	num_heads: number of attention heads
	"""
	# Input layer
	input_layer = Input(shape=shape[1:])  # Remove batch dimension
	
	# Project input to match neurons dimension for residual connections
	x = Dense(neurons)(input_layer)
	
	# Multi-head attention layer
	attention_output = MultiHeadAttention(
		num_heads=num_heads,
		key_dim=neurons // num_heads
	)(x, x)
	
	# Add & Norm
	attention_output = Dropout(dropout_rate)(attention_output)
	x = LayerNormalization(epsilon=1e-6)(x + attention_output)
	
	# Feed-forward network
	ffn_output = Dense(neurons * 2, activation='relu')(x)
	ffn_output = Dropout(dropout_rate)(ffn_output)
	ffn_output = Dense(neurons)(ffn_output)
	
	# Add & Norm
	ffn_output = Dropout(dropout_rate)(ffn_output)
	x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
	
	# Global pooling to reduce sequence dimension
	x = GlobalAveragePooling1D()(x)
	
	# Output layer
	output = Dense(class_count, activation=activation)(x)
	
	# Create model
	model = Model(inputs=input_layer, outputs=output)
	model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
	print(model.summary())
	return model

def get_model(model_type, shape, neurons, stateful, class_count=1, dense_layers=[], loss="mse", optimizer="adam", dual_activation=["linear", "linear"], stacked_layers=2,
				enable_feedback_output=False, regression_activation="linear", num_heads=4, dropout_rate=0.1):
	if model_type == "stacked_lstm":
		model = get_stacked_lstm_model(shape, neurons, stateful, class_count, loss, optimizer, stacked_layers, regression_activation)
	elif model_type == "mlp":
		model = get_mlp_model(shape, neurons, class_count, dense_layers, loss, optimizer, regression_activation)
	elif model_type == "transformer":
		model = get_transformer_model(shape, neurons, num_heads=num_heads, class_count=class_count, loss=loss, optimizer=optimizer, activation=regression_activation, dropout_rate=dropout_rate)
	else:# Default: use single LSTM layer
		model = get_lstm_model(shape, neurons, stateful, class_count, dense_layers, loss, optimizer, regression_activation)
	return model