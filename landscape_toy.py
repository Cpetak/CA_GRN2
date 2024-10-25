import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def gaussian(x, mu=0.0, sigma=1.0):
    """
    Compute the value of a 1D Gaussian function.

    Parameters:
    x: float or array-like
        The input value(s) where to evaluate the Gaussian
    mu: float
        The mean (center) of the Gaussian
    sigma: float
        The standard deviation (width) of the Gaussian

    Returns:
    float or array-like
        The value of the Gaussian function at x
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def square(x):
    return (np.abs(x) < 0.3) * 1.2


# Set random seed for reproducibility
np.random.seed(42)

# Generate random points
N = 100  # Number of points
x = np.linspace(-5, 5, N)
y = np.random.uniform(0, 1, N)
# localopt = gaussian(x, sigma=0.2) / 4.0
# y += localopt
localopt = square(x)
y += y * localopt


# Apply Gaussian smoothing
sigma = 0.8  # Controls the smoothing width
y_smoothed = gaussian_filter1d(y, sigma)

# Create interpolation function
f_interpolated = interp1d(
    x, y_smoothed, kind="cubic", bounds_error=False, fill_value="extrapolate"
)


# Setup the figure and animation
fig, ax = plt.subplots(figsize=(12, 6))
# Calculate scale based on frame number
scale = 0.1
# Generate samples with current scale
M = 10_000
x_samples = np.linspace(-5, 5, M)
y_samples = f_interpolated(x_samples)
# Plot everything
ax.plot(x_samples, y_samples, ".", color="red", markersize=1, label="Sampled Points")
ax.plot(x, localopt, ".", color="red", markersize=1, label="bonus")
# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.grid(True)
# Set consistent axis limits
ax.set_xlim(-5, 5)
ax.set_ylim(-0.5, 2)
plt.tight_layout(
plt.show()


# %%

# Setup the figure and animation
fig, ax = plt.subplots(figsize=(12, 6))


def animate(frame):
    ax.clear()

    # Calculate scale based on frame number
    scale = 0.01 + (frame / 50) * 1.0

    # Generate samples with current scale
    M = 10_000
    x_samples = np.random.randn(M) * scale
    y_samples = f_interpolated(x_samples)

    # Plot everything
    x = np.linspace(-4, 4, 10_000)
    ax.plot(x, f_interpolated(x), alpha=0.5, label="Fitness lanscape")
    ax.plot(x_samples, y_samples, ".", color="red", markersize=1, label="Kids")
    ax.hist(x_samples, density=True, alpha=0.5, bins=50, label="Distribution of Kids")
    ax.plot(
        np.linspace(min(x_samples), max(x_samples), M),
        np.sort(y_samples),
        ".",
        color="green",
        markersize=1,
        label="Kids Fitness",
    )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Gaussian Smoothing Animation (Scale: {scale:.2f})")
    ax.legend()
    ax.grid(True)

    # Set consistent axis limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, 2)


# Create animation
anim = FuncAnimation(fig, animate, frames=50, interval=200, blit=False)

# Uncomment the following lines to save the animation
# anim.save('gaussian_animation.gif', writer='pillow')

plt.tight_layout()
plt.show()


# %%

# Setup the figure and animation
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate scale based on frame number
scale = 0.4

# Generate samples with current scale
M = 10_000
x_samples = np.random.randn(M) * scale
y_samples = f_interpolated(x_samples)

# Plot everything
x = np.linspace(-4, 4, 10_000)
ax.plot(x, f_interpolated(x), alpha=0.5, label="Fitness lanscape")
ax.plot(x_samples, y_samples, ".", color="red", markersize=1, label="Kids")
ax.hist(x_samples, density=True, alpha=0.5, bins=50, label="Distribution of Kids")
ax.plot(
    np.linspace(min(x_samples), max(x_samples), M),
    np.sort(y_samples),
    ".",
    color="green",
    markersize=1,
    label="Kids Fitness",
)
ax.plot(
    np.linspace(min(x_samples), max(x_samples), M),
    gaussian_filter1d(np.gradient(np.sort(y_samples)), sigma=20)*2_000,
    ",",
    color="black",
    label="Grad",
)
# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Gaussian Smoothing Animation (Scale: {scale:.2f})")
ax.legend()
ax.grid(True)

# Set consistent axis limits
ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 2)

plt.tight_layout()
plt.show()

# Uncomment the following lines to save the animation
# anim.save('gaussian_animation.gif', writer='pillow')
