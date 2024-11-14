import tensorflow as tf

# Input layer
input_layer = tf . keras . layers . Input ( shape =(10 ,) )


# Hidden layers
hidden_layer1 = tf.keras.layers.Dense( units =64 , activation =  'relu') ( input_layer)
hidden_layer2 = tf.keras.layers.Dense ( units =128 , activation = 'relu') (hidden_layer1 )
hidden_layer3 = tf.keras. layers . Dense ( units =64 , activation = 'relu') (hidden_layer2 )

# Output layer
output_layer = tf.keras.layers.Dense( units =10 , activation = 'softmax') (hidden_layer3 )

# Model build
model = tf.keras.Model(inputs = input_layer , outputs = output_layer )

# Model compile
model.compile()
