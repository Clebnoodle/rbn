import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess the California Housing dataset
housing = fetch_california_housing()
x, y = housing.data, housing.target

# Standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the dataset into training, validation, and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

# Define the Radial Basis Function
def radial_basis_function(x, centers, beta):
    diff = tf.expand_dims(x, axis=1) - tf.expand_dims(centers, axis=0)
    squared_diff = tf.reduce_sum(tf.square(diff), axis=2)
    return tf.exp(-beta * squared_diff)

# Create the RBN Model Class
class RBN(tf.keras.Model):
    def __init__(self, n_centers, beta, output_dim):
        super(RBN, self).__init__()
        self.n_centers = n_centers
        self.beta = beta
        self.output_dim = output_dim
        self.centers = tf.Variable(tf.random.normal([n_centers, x_train.shape[1]]), trainable=True)
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        rbf_output = radial_basis_function(inputs, self.centers, self.beta)
        return self.output_layer(rbf_output)

# Parameters for California Housing dataset
n_centers = 120
beta = 0.35
output_dim = 1  # Regression output

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
#     ModelCheckpoint("Bst_model.h5", save_best_only=True)
]

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.003
)

# Instantiate and compile the model
model = RBN(n_centers, beta, output_dim)
model.compile(optimizer=optimizer, loss='mse')

# Training the model
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=callbacks)

# Save the model weights
model.save_weights('rbf_save.h5')

# Evaluating the model
loss = model.evaluate(x_test, y_test)
print(f'Mean Squared Error: {loss:.2f}')

# Predicting and visualizing the results
y_pred = model.predict(x_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2:.2f}')

plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('California Housing Predictions vs. True Values')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.show()
