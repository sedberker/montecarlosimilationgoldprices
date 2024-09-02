import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Monte Carlo simulation parameters
mu = 0.000202198  # Average daily logarithmic return calculated since 01.01.1979
sigma = 0.011643114  # Standard deviation of daily logarithmic return calculated since 01.01.1979
days = 252  # Time period (1 year = 252 trading days)
simulations = 1000  # Number of simulations
initial_price = 2511.20  # Current gold price (closing price on 23.08.2024)

# Generating simulations
price_paths = np.zeros((simulations, days))
price_paths[:, 0] = initial_price

# Calculating price paths using log-normal distribution
for i in range(1, days):
    daily_returns = np.random.lognormal(mu, sigma, simulations)
    price_paths[:, i] = price_paths[:, i-1] * daily_returns

# Calculating logarithmic returns on the 252nd day
final_log_returns = np.log(price_paths[:, -1] / initial_price)

# Calculating confidence intervals (2.5% and 97.5% percentiles)
percentile_2_5 = np.percentile(price_paths, 2.5, axis=0)
percentile_97_5 = np.percentile(price_paths, 97.5, axis=0)

# Calculating average and median price paths
average_price_path = np.mean(price_paths, axis=0)
median_price_path = np.median(price_paths, axis=0)

# Get the average, median, min, and max prices on the 252nd day
final_average_price = average_price_path[-1]
final_median_price = median_price_path[-1]
final_min_price = np.min(price_paths[:, -1])
final_max_price = np.max(price_paths[:, -1])

# 1st Plot: Price Paths
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, simulations))
for i in range(simulations):
    plt.plot(price_paths[i], lw=1, color=colors[i])

# Plotting confidence intervals and price paths
plt.plot(percentile_2_5, color='black', lw=2, linestyle='--', label=f'Lower Confidence Interval (End: ${percentile_2_5[-1]:.2f})')
plt.plot(percentile_97_5, color='black', lw=2, linestyle='--', label=f'Upper Confidence Interval (End: ${percentile_97_5[-1]:.2f})')
plt.plot(average_price_path, color='black', lw=2.5, label=f"Average Price Path (End: ${final_average_price:.2f})")
plt.plot(median_price_path, color='orange', lw=2.5, label=f"Median Price Path (End: ${final_median_price:.2f})")
plt.plot([], [], ' ', label=f"Min Price (End: ${final_min_price:.2f})")
plt.plot([], [], ' ', label=f"Max Price (End: ${final_max_price:.2f})")

# Plot titles and labels
plt.title("Gold Price Monte Carlo Simulation (1 Year) and 95% Confidence Interval")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend(loc='upper left')
plt.show()

# 2nd Plot: Histogram of Logarithmic Returns on the 252nd Day
plt.figure(figsize=(10, 6))
plt.hist(final_log_returns, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Logarithmic Returns on the 252nd Day")
plt.xlabel("Logarithmic Return")
plt.ylabel("Frequency")
plt.show()

# 3rd Plot: Histogram of Prices on the 252nd Day and Lognormal Probability Density Function
final_prices = price_paths[:, -1]

plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(final_prices, bins=50, color='blue', edgecolor='black', alpha=0.7, density=True)

# Plotting the log-normal probability density function
shape, loc, scale = lognorm.fit(final_prices, floc=0)
pdf = lognorm.pdf(bins, shape, loc, scale)
plt.plot(bins, pdf, 'r-', lw=2, label='Lognormal Probability Density Function')

# Plot titles and labels
plt.title("Distribution of Simulated Gold Prices on the 252nd Day")
plt.xlabel("Price (USD)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
