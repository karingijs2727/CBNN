import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(num_samples, num_clusters=2, coord_dimensions=2):
    t = np.linspace(0, 10, num_samples)
    data = np.zeros((num_samples, coord_dimensions))
    data[:, 0] = t
    data[:, 1] = np.sin(t)  # Generating data points along a sine wave
    labels = np.zeros(num_samples)  # For demonstration purposes, no actual clustering needed
    return data, labels

def generate_dataAbove_zero(num_samples, num_clusters=2, coord_dimensions=2):
    t = np.linspace(0, 10, num_samples)
    data = np.zeros((num_samples, coord_dimensions))
    data[:, 0] = t
    data[:, 1] = 0.5 * (np.sin(t) + 1)  # Scale sine wave to range [0, 1]
    labels = np.zeros(num_samples)  # For demonstration purposes, no actual clustering needed
    return data, labels


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#BEST NOW
def squared_euclidean_distance(x, y):
    return np.sum((x - y) ** 2, axis=-1)

def squared_euclidean_distanceASDaA(x, y):
    return np.sum(np.abs(x - y), axis=-1)

def squared_euclidean_distanceASD(x, y):
    return np.linalg.norm(x - y, axis=-1) ** 2





class CoordinatesNeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs, coord_dimensions=3, boundary=10.0):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.coord_dimensions = coord_dimensions
        self.boundary = boundary
        
        # Initialize coordinates for each neuron
        self.input_neurons = np.random.rand(num_inputs, coord_dimensions) * boundary * 2 - boundary
        self.hidden_neurons = np.random.rand(num_hidden, coord_dimensions) * boundary * 2 - boundary
        self.output_neurons = np.random.rand(num_outputs, coord_dimensions) * boundary * 2 - boundary
    
    def enforce_boundaries(self):
        # Ensure neurons stay within boundaries
        self.input_neurons = np.clip(self.input_neurons, -self.boundary, self.boundary)
        self.hidden_neurons = np.clip(self.hidden_neurons, -self.boundary, self.boundary)
        self.output_neurons = np.clip(self.output_neurons, -self.boundary, self.boundary)

    def forward(self, inputs):
        batch_size = len(inputs)
        
        hidden_activations = np.zeros((batch_size, self.num_hidden))
        output_activations = np.zeros((batch_size, self.num_outputs))
        
        # Calculate activations from inputs to hidden layer
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                distance_squared = squared_euclidean_distance(inputs[:, np.newaxis, i], self.input_neurons[i])
                hidden_activations[:, j] += 1.0 / (distance_squared + 1e-6)
        
        hidden_activations = np.clip(hidden_activations, 0, 1e6)
        
        # Calculate activations from hidden layer to output layer
        for j in range(self.num_hidden):
            for k in range(self.num_outputs):
                distance_squared = squared_euclidean_distance(self.hidden_neurons[j], self.output_neurons[k])
                output_activations[:, k] += hidden_activations[:, j] / (distance_squared + 1e-6)
        
        return output_activations
    
    def backward(self, inputs, targets, learning_rate=0.1):
        outputs = self.forward(inputs)
        loss = 0.5 * np.mean((outputs - targets) ** 2)
        #loss = np.mean((targets - outputs) ** 2)
        #loss = np.mean((targets - outputs))
        
        input_gradients = np.zeros_like(self.input_neurons)
        hidden_gradients = np.zeros_like(self.hidden_neurons)
        output_gradients = np.zeros_like(self.output_neurons)
        
        # Compute distances
        input_to_hidden_distances_squared = squared_euclidean_distance(self.input_neurons[:, np.newaxis], self.hidden_neurons)
        hidden_to_output_distances_squared = squared_euclidean_distance(self.hidden_neurons[:, np.newaxis], self.output_neurons)
        
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                for k in range(self.num_outputs):
                    output_gradients[k] += (outputs[k] - targets[k]) / (hidden_to_output_distances_squared[j, k] ** 3) * (self.hidden_neurons[j] - self.output_neurons[k])
                    hidden_gradients[j] += (outputs[k] - targets[k]) / (hidden_to_output_distances_squared[j, k] ** 3) * (self.output_neurons[k] - self.hidden_neurons[j])
                    input_gradients[i] += (outputs[k] - targets[k]) / (hidden_to_output_distances_squared[j, k] ** 3) * (self.hidden_neurons[j] - self.input_neurons[i])
        
        # Update neuron weights
        self.hidden_neurons -= learning_rate * hidden_gradients
        
        # Ensure neurons stay within boundaries
        self.enforce_boundaries()
        
        return loss

import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_samples = 100
    num_clusters = 2
    coord_dimensions = 10
    
    # Generate synthetic data
    #data, labels = generate_data(num_samples, num_clusters, coord_dimensions)
    data, labels = generate_data(num_samples, num_clusters, 2)
    
    nn = CoordinatesNeuralNetwork(2, 12, 1, coord_dimensions=3, boundary=1.5)

    # Example input and target
    inputs = data
    targets = data[:, 1].reshape(-1, 1)  # Predicting the sine wave values

    #inputs = np.array([[0.2, 0.2], [0.4, 0.4]])
    #targets = np.array([[0.3], [0.6]])  # Two target values corresponding to two output neurons

    #inputs = np.array([[0.2, 0.2]])
    #targets = np.array([[0.3]])  # Two target values corresponding to two output neurons

    num_epochs = 10000
    learning_rate = 0.001
    
    for epoch in range(num_epochs):
        loss = nn.backward(inputs, targets, learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    final_outputs = nn.forward(inputs)
    print("Final Output activations (first 10):", final_outputs[:2])


# Plot the original sine wave and predicted sine wave
fig, ax = plt.subplots(figsize=(10, 6))

# Plot original sine wave
ax.plot(data[:, 0], data[:, 1], label='Original Sine Wave')

# Plot predicted sine wave
ax.plot(data[:, 0], final_outputs.flatten(), label='Predicted Sine Wave', linestyle='--')

ax.set_xlabel('X (Time)')
ax.set_ylabel('Y (Sine Value)')
ax.set_title('Original vs Predicted Sine Wave')
ax.legend()

plt.tight_layout()
plt.show()
