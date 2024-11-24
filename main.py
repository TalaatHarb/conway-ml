import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import matplotlib.pyplot as plt


def generate_grid(size, prob_alive=0.3):
    """Generate a random grid of given size with a probability of cells being alive."""
    return (np.random.rand(size, size) < prob_alive).astype(int)

def next_state(grid):
    """Compute the next state of the grid using Conway's Game of Life rules."""
    size = grid.shape[0]
    next_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            # Count live neighbors
            live_neighbors = sum(
                grid[(i + x) % size, (j + y) % size]
                for x in [-1, 0, 1]
                for y in [-1, 0, 1]
                if not (x == 0 and y == 0)
            )
            # Apply rules
            if grid[i, j] == 1 and live_neighbors in [2, 3]:
                next_grid[i, j] = 1
            elif grid[i, j] == 0 and live_neighbors == 3:
                next_grid[i, j] = 1
    return next_grid

# Generate dataset
def generate_dataset(num_samples, grid_size):
    dataset = []
    for _ in range(num_samples):
        grid = generate_grid(grid_size)
        next_grid = next_state(grid)
        dataset.append((next_grid, grid))
    return dataset

class GameOfLifeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        next_grid, current_grid = self.dataset[idx]
        return torch.tensor(next_grid, dtype=torch.float32), torch.tensor(current_grid, dtype=torch.float32)

class ReverseCNN(nn.Module):
    def __init__(self):
        super(ReverseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x

# Training the model
def train_model(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for next_grid, current_grid in dataloader:
            next_grid = next_grid.unsqueeze(1)  # Add channel dimension
            current_grid = current_grid.unsqueeze(1)
            output = model(next_grid)
            loss = criterion(output, current_grid)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        
def visualize_sequence(sequence):
    """Visualize a sequence of grids and navigate using arrow keys."""
    fig, ax = plt.subplots()
    idx = [0]  # Use a list to make it mutable

    def update_plot(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(sequence)
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(sequence)
        ax.clear()
        ax.imshow(sequence[idx[0]], cmap='binary')
        ax.set_title(f"Step {idx[0]}")
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', update_plot)
    ax.imshow(sequence[0], cmap='binary')
    ax.set_title("Step 0")
    plt.show()

def evaluate_model(model, dataloader):
    """
    Evaluate the model using grid accuracy and mean squared error.

    Parameters:
    - model: The trained CNN model.
    - dataloader: DataLoader for test data.

    Returns:
    - average_accuracy: The average grid accuracy over all test samples.
    - average_mse: The average mean squared error over all test samples.
    """
    total_accuracy = 0
    total_mse = 0
    total_samples = 0

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for next_grid, current_grid in dataloader:
            next_grid = next_grid.unsqueeze(1)  # Add channel dimension
            current_grid = current_grid.unsqueeze(1)

            # Predict using the model
            predicted_current_grid = model(next_grid)

            # Binary predictions
            predicted_binary = (predicted_current_grid > 0.5).float()

            # Compute metrics
            batch_accuracy = (predicted_binary == current_grid).float().mean().item()
            batch_mse = F.mse_loss(predicted_current_grid, current_grid).item()

            # Aggregate metrics
            total_accuracy += batch_accuracy * next_grid.size(0)  # Multiply by batch size
            total_mse += batch_mse * next_grid.size(0)
            total_samples += next_grid.size(0)

    average_accuracy = total_accuracy / total_samples
    average_mse = total_mse / total_samples

    return average_accuracy, average_mse


# Step 1: Generate the dataset
print("Generating dataset...")
num_samples = 1000  # Number of samples in the dataset
grid_size = 10      # Size of the grids (10x10)
dataset = generate_dataset(num_samples, grid_size)
print("Dataset generation complete.")

# Step 2: Prepare DataLoader
batch_size = 32
game_of_life_dataset = GameOfLifeDataset(dataset)
dataloader = DataLoader(game_of_life_dataset, batch_size=batch_size, shuffle=True)

# Step 3: Initialize and train the model
print("Initializing and training the CNN model...")
model = ReverseCNN()
epochs = 50
learning_rate = 0.001
train_model(model, dataloader, epochs=epochs, lr=learning_rate)
print("Model training complete.")

# Step 4: Test the model (optional)
# Pick a sample from the dataset
test_next_grid, test_current_grid = generate_dataset(num_samples, grid_size)[0]
test_next_grid_tensor = torch.tensor(test_next_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_current_grid = model(test_next_grid_tensor).detach().squeeze().numpy()
predicted_current_grid_binary = (predicted_current_grid > 0.5).astype(int)

# Display the original and predicted grids
print("\nTest Case Results:")
print("Next Grid (Input):")
print(test_next_grid)
print("True Current Grid:")
print(test_current_grid)
print("Predicted Current Grid:")
print(predicted_current_grid_binary)

# Step 5: Visualize a sequence of grids
print("Visualizing a sequence of grids...")
sequence = [item[0] for item in dataset[:10]]  # Extract first 10 "next states" as a sequence
visualize_sequence(sequence)

# Step 6: Evaluate the model on the test set
print("Evaluating the model...")
test_dataloader = DataLoader(game_of_life_dataset, batch_size=batch_size, shuffle=False)
accuracy, mse = evaluate_model(model, test_dataloader)

print("\nModel Evaluation Metrics:")
print(f"Grid Accuracy: {accuracy:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
