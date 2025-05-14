import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

D = [-200, -100, -170, 15]
a = [-1, -1, -6.5, 0.7]
b = [0, 0, 11, 0.6]
c = [-10, -10, -6.5, 0.7]
X = [1, 0, -0.5, -1]
Y = [0, 0.5, 1.5, 1]
sigma = 0.05

def mueller_potential(x1, x2, gamma=0):
    """
    Compute the Mueller potential without roughness (gamma=0)
    
    Parameters:
    x1, x2: Coordinates, can be scalars or arrays
    gamma: Roughness parameter (set to 0 as per instructions)
    
    Returns:
    V: Potential energy
    """
    value = 0
    for i in range(4):
        value += D[i] * np.exp(a[i] * (x1 - X[i])**2 + 
                                b[i] * (x1 - X[i]) * (x2 - Y[i]) + 
                                c[i] * (x2 - Y[i])**2)
    return value

def mueller_potential_torch(x, gamma=0):
    """
    Compute the Mueller potential for PyTorch tensors
    
    Parameters:
    x: Tensor of shape [batch_size, 10]
    gamma: Roughness parameter (set to 0)
    
    Returns:
    V: Potential energy tensor
    """
    x1, x2 = x[:, 0], x[:, 1]
    value = torch.zeros_like(x1)
    
    for i in range(4):
        value += D[i] * torch.exp(a[i] * (x1 - X[i])**2 + 
                                  b[i] * (x1 - X[i]) * (x2 - Y[i]) + 
                                  c[i] * (x2 - Y[i])**2)
    value += 1/(2*sigma**2) * torch.sum(x[:,2:]**2, dim=1)
    value += 0.05*torch.sin(10*torch.pi*x1)*torch.sin(10*torch.pi*x2)
    return value

def compute_gradient(x, potential_fn):
    """
    Compute gradient of the potential with respect to coordinates
    
    Parameters:
    x: Tensor of shape [batch_size, 2]
    potential_fn: Function that computes the potential
    
    Returns:
    grad_V: Gradient tensor of shape [batch_size, 2]
    """
    x_with_grad = x.detach().clone().requires_grad_(True)
    V = potential_fn(x_with_grad)
    
    grad_outputs = torch.ones_like(V)
    grad_V, = torch.autograd.grad(V, x_with_grad, grad_outputs=grad_outputs, create_graph=True)
    
    return grad_V

def langevin_dynamics(x0, n_steps, dt, beta, potential_fn=mueller_potential_torch):
    """
    Simulate overdamped Langevin dynamics using Euler-Maruyama scheme
    
    Parameters:
    x0: Initial position tensor of shape [batch_size, 10]
    n_steps: Number of time steps
    dt: Time step size
    beta: Inverse temperature (1/kBT)
    potential_fn: Function that computes the potential
    
    Returns:
    traj: Trajectory tensor of shape [batch_size, n_steps+1, 10]
    """
    batch_size = x0.shape[0]
    traj = torch.zeros((batch_size, n_steps+1, 10), device=x0.device)
    traj[:, 0, :] = x0
    
    for i in range(n_steps):
        x = traj[:, i, :]
        grad_V = compute_gradient(x, potential_fn)
        noise = torch.randn_like(x) * np.sqrt(2 * dt / beta)
        x_new = x - grad_V * dt + noise
        traj[:, i+1, :] = x_new
    
    return traj

def metadynamics(x0, n_steps, dt, beta, tau=500, w=5, sigma=0.05, potential_fn=mueller_potential_torch):
    """
    Simulate metadynamics to fill potential wells
    
    Parameters:
    x0: Initial position tensor of shape [batch_size, 10]
    n_steps: Number of time steps
    dt: Time step size
    beta: Inverse temperature (1/kBT)
    tau: Frequency of Gaussian deposition
    w: Height of Gaussian
    sigma: Width of Gaussian
    potential_fn: Function that computes the potential
    
    Returns:
    traj: Trajectory tensor of shape [batch_size, n_steps+1, 10]
    V_G_values: Tensor of shape [batch_size, n_steps+1] containing the bias values
    """
    batch_size = x0.shape[0]
    traj = torch.zeros((batch_size, n_steps+1, 10), device=x0.device)
    V_G_values = torch.zeros((batch_size, n_steps+1), device=x0.device)
    traj[:, 0, :] = x0
    gaussian_centers = []
    gaussian_indices = []
    
    for i in range(n_steps):
        x = traj[:, i, :]
        grad_V = compute_gradient(x, potential_fn)
        grad_VG = torch.zeros_like(x)
        current_VG = torch.zeros(batch_size, device=x.device)
        
        for center, idx in zip(gaussian_centers, gaussian_indices):
            dist = x[:, :2] - center[:, :2]
            grad_gaussian = -dist * w * torch.exp(-torch.sum(dist**2, dim=1, keepdim=True) / (2*sigma**2)) / sigma**2
            grad_VG[:, :2] += grad_gaussian
            dist2 = torch.sum(dist**2, dim=1)
            current_VG += w * torch.exp(-dist2 / (2*sigma**2))
        V_G_values[:, i] = current_VG
        noise = torch.randn_like(x) * np.sqrt(2 * dt / beta)
        x_new = x - (grad_V + grad_VG) * dt + noise
        traj[:, i+1, :] = x_new
        if (i+1) % tau == 0:
            gaussian_centers.append(x_new.detach().clone())
            gaussian_indices.append(i+1)
    for center, idx in zip(gaussian_centers, gaussian_indices):
        dist = traj[:, -1, :2] - center[:, :2]
        dist2 = torch.sum(dist**2, dim=1)
        V_G_values[:, -1] += w * torch.exp(-dist2 / (2*sigma**2))
    return traj, V_G_values

def create_mollifier(center, radius, eps=0.02):
    """
    Create a mollifier function that transitions smoothly from 1 to 0
    
    Parameters:
    center: Center of the region
    radius: Radius of the region
    eps: Width of the transition region
    
    Returns:
    chi: Mollifier function
    """
    def chi(x):
        center_tensor = torch.tensor(center, device=x.device)
        dist2 = torch.sum((x[:, :2] - center_tensor)**2, dim=1)
        return 0.5 - 0.5 * torch.tanh(1000 * (dist2 - (radius + eps)**2))
    
    return chi

class CommittorNetwork(nn.Module):
    """
    Neural network to approximate the committor function
    """
    def __init__(self, input_dim, hidden_layers):
        super(CommittorNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_committor(model, optimizer, data_loader, chi_A, chi_B, beta, beta_prime=None, V_G=None, n_epochs=100):
    """
    Train the committor function neural network
    
    Parameters:
    model: Neural network model
    optimizer: Optimizer
    data_loader: DataLoader with training data
    chi_A, chi_B: Mollifier functions
    beta: Inverse temperature (1/kBT)
    beta_prime: Inverse temperature for importance sampling (if None, use metadynamics)
    V_G: Modified potential from metadynamics (if beta_prime is None)
    n_epochs: Number of training epochs
    
    Returns:
    losses: List of training losses
    """
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(device)
            x_batch.requires_grad_(True)
            q_tilde = model(x_batch)
            
            q = (1 - chi_A(x_batch)) * ((1 - chi_B(x_batch)) * q_tilde + chi_B(x_batch))
            
            grad_q = torch.autograd.grad(q.sum(), x_batch, create_graph=True)[0]
            
            if beta_prime is not None:
                likelihood_ratio = torch.exp(-(beta - beta_prime) * mueller_potential_torch(x_batch))
                loss = torch.mean(torch.sum(grad_q**2, dim=1) * likelihood_ratio)
            else:
                bias_factor = torch.exp(beta * V_G(x_batch))
                loss = torch.mean(torch.sum(grad_q**2, dim=1) * bias_factor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def train_committor_with_VG(model, optimizer, data_loader, chi_A, chi_B, beta, n_epochs=100):
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x_batch, V_G_batch in data_loader:
            x_batch = x_batch.to(device)
            V_G_batch = V_G_batch.to(device)
            x_batch.requires_grad_(True)
            
            q_tilde = model(x_batch)
            
            q = (1 - chi_A(x_batch)) * ((1 - chi_B(x_batch)) * q_tilde + chi_B(x_batch))
            
            grad_q = torch.autograd.grad(q.sum(), x_batch, create_graph=True)[0]
            
            bias_factor = torch.exp(beta * V_G_batch)
            loss = torch.mean(torch.sum(grad_q**2, dim=1) * bias_factor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def sample_transition_states(model, chi_A, chi_B, n_samples=100, n_steps=1000, dt=1e-5, kappa=3e4, beta=0.1):
    """
    Sample transition states by constraining dynamics to q=0.5 surface
    
    Parameters:
    model: Trained neural network model
    chi_A, chi_B: Mollifier functions
    n_samples: Number of samples to generate
    n_steps: Number of steps per sample
    dt: Time step
    kappa: Constraint strength
    beta: Inverse temperature
    
    Returns:
    states: Sampled transition states
    """
    x = torch.randn(n_samples, 10).to(device)
    
    for i in range(n_steps):
        x_with_grad = x.detach().clone().requires_grad_(True)
        q_tilde = model(x_with_grad)
        q = (1 - chi_A(x_with_grad)) * ((1 - chi_B(x_with_grad)) * q_tilde + chi_B(x_with_grad))
        
        grad_V = compute_gradient(x_with_grad, mueller_potential_torch)
        
        constraint = 0.5 * kappa * (q - 0.5)**2
        grad_constraint = torch.autograd.grad(constraint.sum(), x_with_grad)[0]
        
        x_with_grad.requires_grad_(False)
        
        noise = torch.randn_like(x) * np.sqrt(2 * dt / beta)
        
        x = x - (grad_V + grad_constraint) * dt + noise
    
    return x.detach().cpu().numpy()

def visualize_potential_and_committor(model, chi_A, chi_B, x_range=(-1.5, 1), y_range=(-0.5, 2), resolution=100):
    """
    Visualize the potential energy surface and the committor function
    
    Parameters:
    model: Trained neural network model
    chi_A, chi_B: Mollifier functions
    x_range, y_range: Range for visualization
    resolution: Grid resolution
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z_potential = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z_potential[i, j] = mueller_potential(X[i, j], Y[i, j])
    
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    grid_points_padded = np.pad(grid_points, ((0, 0), (0, 8)), mode='constant', constant_values=0)
    grid_tensor = torch.tensor(grid_points_padded, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        q_tilde = model(grid_tensor)
        q = (1 - chi_A(grid_tensor)) * ((1 - chi_B(grid_tensor)) * q_tilde + chi_B(grid_tensor))
    
    Z_committor = q.cpu().numpy().reshape(X.shape)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    contour1 = ax1.contour(X, Y, Z_potential, 20, colors='k', linewidths=0.5)
    cf1 = ax1.contourf(X, Y, Z_potential, 50, cmap='viridis')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('Mueller Potential')
    plt.colorbar(cf1, ax=ax1, label='Energy')
    
    contour2 = ax2.contour(X, Y, Z_committor, levels=[0.1, 0.5, 0.9], colors='k', linewidths=1.5)
    cf2 = ax2.contourf(X, Y, Z_committor, 50, cmap='coolwarm')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Committor Function')
    plt.colorbar(cf2, ax=ax2, label='Committor Value')
    ax2.clabel(contour2, inline=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('potential_and_committor.png', dpi=300)
    plt.close()
    
    return

def find_minima():
    """
    Find the local minima of the Mueller potential
    
    Returns:
    minima: List of (x, y) coordinates of local minima
    """
    from scipy.optimize import minimize
    
    def opt_potential(coords):
        return mueller_potential(coords[0], coords[1])
    
    initial_guesses = [
        (-0.5, 1.5),
        (0.6, 0.0)
    ]
    
    minima = []
    for guess in initial_guesses:
        result = minimize(opt_potential, guess, method='BFGS')
        if result.success:
            minima.append((result.x[0], result.x[1]))   
    return minima

def compare_sampling_methods(model_higher_temp, model_meta, chi_A, chi_B, x_range=(-1.5, 1), y_range=(-0.5, 2), resolution=100):
    """
    Compare and visualize committor functions from different sampling methods
    
    Parameters:
    model_higher_temp: Model trained with artificial temperature
    model_meta: Model trained with metadynamics
    chi_A, chi_B: Mollifier functions
    x_range, y_range: Domain boundaries
    resolution: Grid resolution
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    grid_points_padded = np.pad(grid_points, ((0, 0), (0, 8)), mode='constant', constant_values=0)
    grid_tensor = torch.tensor(grid_points_padded, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        q_tilde_temp = model_higher_temp(grid_tensor)
        q_temp = (1 - chi_A(grid_tensor)) * ((1 - chi_B(grid_tensor)) * q_tilde_temp + chi_B(grid_tensor))
        q_temp = q_temp.cpu().numpy().reshape(X.shape)
        
        q_tilde_meta = model_meta(grid_tensor)
        q_meta = (1 - chi_A(grid_tensor)) * ((1 - chi_B(grid_tensor)) * q_tilde_meta + chi_B(grid_tensor))
        q_meta = q_meta.cpu().numpy().reshape(X.shape)
    
    diff = np.abs(q_temp - q_meta)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    im1 = ax1.contourf(X, Y, q_temp, 50, cmap='coolwarm')
    ax1.set_title('Artificial Temperature Solution')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.contourf(X, Y, q_meta, 50, cmap='coolwarm')
    ax2.set_title('Metadynamics Solution')
    plt.colorbar(im2, ax=ax2)
    
    im3 = ax3.contourf(X, Y, diff, 50, cmap='viridis')
    ax3.set_title(f'Difference (max: {max_diff:.3f}, mean: {mean_diff:.3f})')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('sampling_methods_comparison.png', dpi=300)
    plt.close()
    
    return max_diff, mean_diff

def main():
    minima = find_minima()
    print(f"Local minima of the Mueller potential: {minima}")
    
    min_A = minima[0]
    min_B = minima[1]
    
    radius = 0.1
    
    chi_A = create_mollifier(min_A, radius)
    chi_B = create_mollifier(min_B, radius)
    
    kB_T = 10
    beta = 1.0 / kB_T
    
    kB_T_prime = 20
    beta_prime = 1.0 / kB_T_prime
    
    dt = 1e-5
    n_steps = 100000    
    print("Generating data using artificial temperature...")
    
    batch_size = 100
    x0 = torch.tensor(np.random.uniform(-1.5, 1.0, (batch_size, 1)), dtype=torch.float32).to(device)
    y0 = torch.tensor(np.random.uniform(-0.5, 2.0, (batch_size, 1)), dtype=torch.float32).to(device)
    z0 = torch.tensor(np.random.uniform(-1, 1, (batch_size, 8)), dtype=torch.float32).to(device)
    x0 = torch.cat([x0, y0, z0], dim=1)
    
    traj_higher_temp = langevin_dynamics(x0, n_steps, dt, beta_prime)
    stride = 100
    samples_higher_temp = traj_higher_temp[:, ::stride, :].reshape(-1,10)
    
    with torch.no_grad():
        mask_A = chi_A(samples_higher_temp) < 0.5
        mask_B = chi_B(samples_higher_temp) < 0.5
        mask = mask_A & mask_B
        samples_higher_temp = samples_higher_temp[mask]
    
    print(f"Collected {len(samples_higher_temp)} samples at higher temperature")
    
    print("Generating data using metadynamics...")
    
    traj_meta, V_G_values = metadynamics(x0, n_steps // 10, dt, beta)
    
    samples_meta = traj_meta[:, ::stride, :].reshape(-1, 10)
    V_G_samples = V_G_values[:, ::stride].reshape(-1)
    
    with torch.no_grad():
        mask_A = chi_A(samples_meta) < 0.5
        mask_B = chi_B(samples_meta) < 0.5
        mask = mask_A & mask_B
        samples_meta = samples_meta[mask]
        V_G_samples = V_G_samples[mask]
    
    print(f"Collected {len(samples_meta)} samples using metadynamics")
    
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset_higher_temp = TensorDataset(samples_higher_temp)
    dataloader_higher_temp = DataLoader(dataset_higher_temp, batch_size=256, shuffle=True)
    
    dataset_meta = TensorDataset(samples_meta, V_G_samples)
    dataloader_meta = DataLoader(dataset_meta, batch_size=256, shuffle=True)
    
    
    print("\nTraining model with artificial temperature data...")
    model_higher_temp = CommittorNetwork(input_dim=10, hidden_layers=[20, 20]).to(device)
    optimizer_higher_temp = optim.Adam(model_higher_temp.parameters(), lr=0.001)
    
    losses_higher_temp = train_committor(
        model_higher_temp, 
        optimizer_higher_temp, 
        dataloader_higher_temp, 
        chi_A, chi_B, 
        beta=beta, 
        beta_prime=beta_prime,
        n_epochs=200
    )
    
    print("\nTraining model with metadynamics data...")
    model_meta = CommittorNetwork(input_dim=10, hidden_layers=[20, 20]).to(device)
    optimizer_meta = optim.Adam(model_meta.parameters(), lr=0.001)
    
    losses_meta = train_committor_with_VG(
        model_meta, 
        optimizer_meta, 
        dataloader_meta, 
        chi_A, chi_B, 
        beta=beta,
        n_epochs=200
    )
    
    print("\nGenerating visualizations...")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_higher_temp, label='Artificial Temperature')
    plt.plot(losses_meta, label='Metadynamics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig('training_loss.png', dpi=300)
    plt.close()
    
    visualize_potential_and_committor(model_higher_temp, chi_A, chi_B)
    
    print("\nSampling transition states...")
    transition_states_higher_temp = sample_transition_states(model_higher_temp, chi_A, chi_B)
    
    plt.figure(figsize=(8, 6))
    
    x = np.linspace(-1.5, 1, 100)
    y = np.linspace(-0.5, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = mueller_potential(X[i, j], Y[i, j])
    
    plt.contour(X, Y, Z, 20, colors='k', linewidths=0.5)
    
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    grid_points_padded = np.pad(grid_points, ((0, 0), (0, 8)), mode='constant', constant_values=0)
    grid_tensor = torch.tensor(grid_points_padded, dtype=torch.float32).to(device)


    with torch.no_grad():
        q_tilde = model_higher_temp(grid_tensor)
        q = (1 - chi_A(grid_tensor)) * ((1 - chi_B(grid_tensor)) * q_tilde + chi_B(grid_tensor))
    
    Z_committor = q.cpu().numpy().reshape(X.shape)
    
    committor_contour = plt.contour(X, Y, Z_committor, levels=[0.5], colors='r', linewidths=2)
    plt.clabel(committor_contour, inline=True, fontsize=10, fmt='%1.1f')
    
    plt.scatter(transition_states_higher_temp[:, 0], transition_states_higher_temp[:, 1], 
                c='blue', s=20, alpha=0.5, label='Sampled Transition States')
    
    plt.scatter([min_A[0], min_B[0]], [min_A[1], min_B[1]], c='green', s=100, marker='*', label='Minima')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Mueller Potential with Transition States')
    plt.legend()
    plt.savefig('transition_states.png', dpi=300)
    plt.close()
    
    print("\nComparing different sampling methods...")
    max_diff, mean_diff = compare_sampling_methods(model_higher_temp, model_meta, chi_A, chi_B)
    print(f"Maximum difference between methods: {max_diff:.6f}")
    print(f"Mean difference between methods: {mean_diff:.6f}")
    
    print("\nDone! Results saved as images.")

if __name__ == "__main__":
    main()