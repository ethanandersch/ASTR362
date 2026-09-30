#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# Set working directory to your Lab0 folder
os.chdir("/mnt/c/Users/Ethan/Downloads/Lab0/Lab0")

# Load a single data file
data = np.loadtxt(
    "different_niter/n1000_t10ms/n1000_t10ms_001.dat"
)

# Question 6: Mean and standard deviation
data_mean = np.mean(data)
data_std = np.std(data)

print("Mean =", data_mean)
print("Standard Deviation =", data_std)

# Question 7: Range of values
print("Minimum count =", np.min(data))
print("Maximum count =", np.max(data))

# Question 8: Poisson distribution function
def poisson(mu, nu):
    return np.exp(-mu) * mu**nu / factorial(nu)

# Question 9: Gaussian distribution function
def gaussian(mean, std, x):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -((x - mean)**2) / (2 * std**2)
    )

# Question 10: Smooth theoretical x values
x = np.linspace(np.min(data), np.max(data), 100)

# Question 11: Create theoretical distributions
smooth_poisson = poisson(data_mean, x)
smooth_gaussian = gaussian(data_mean, data_std, x)

# Question 12: Plot histogram and unscaled distributions
plt.figure(figsize=(8, 5))

plt.hist(data, bins=np.arange(np.min(data), np.max(data)+2),
         alpha=0.6, label="Data")

plt.plot(x, smooth_poisson, 'r-', lw=2, label="Poisson")
plt.plot(x, smooth_gaussian, 'g-', lw=2, label="Gaussian")

plt.xlabel("Counts")
plt.ylabel("Frequency")
plt.title("Unscaled Distributions")
plt.legend()
plt.show()

# Question 13: Scale distributions to histogram
bins = np.arange(np.min(data), np.max(data)+2)
bin_width = bins[1] - bins[0]

poisson_scaled = smooth_poisson * len(data) * bin_width
gaussian_scaled = smooth_gaussian * len(data) * bin_width

plt.figure(figsize=(8, 5))

plt.hist(data, bins=bins, alpha=0.6, label="Data")

plt.plot(x, poisson_scaled, 'r-', lw=2,
         label="Scaled Poisson")

plt.plot(x, gaussian_scaled, 'g-', lw=2,
         label="Scaled Gaussian")

plt.xlabel("Counts")
plt.ylabel("Frequency")
plt.title("Scaled Distributions")
plt.legend()
plt.show()


# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# ---------------------------
# SET YOUR DATA LOCATION HERE
# ---------------------------

os.chdir("/mnt/c/Users/Ethan/Downloads/Lab0/Lab0")

# Load one data file
data = np.loadtxt(
    "different_niter/n1000_t10ms/n1000_t10ms_001.dat"
)

# ---------------------------
# Question 6
# ---------------------------

data_mean = np.mean(data)
data_std = np.std(data)

print("Mean =", data_mean)
print("Standard Deviation =", data_std)

# ---------------------------
# Question 7
# ---------------------------

data_min = int(np.min(data))
data_max = int(np.max(data))

print("Minimum Count =", data_min)
print("Maximum Count =", data_max)

# ---------------------------
# Question 8
# ---------------------------

def poisson(mu, nu_values):
    probs = []

    for n in nu_values:
        p = (np.exp(-mu) * mu**n) / factorial(int(n))
        probs.append(p)

    return np.array(probs)

# ---------------------------
# Question 9
# ---------------------------

def gaussian(mean, std, x):
    return (
        1 / (std * np.sqrt(2 * np.pi))
    ) * np.exp(
        -((x - mean) ** 2) / (2 * std**2)
    )

# ---------------------------
# Question 10
# ---------------------------

x = np.arange(data_min, data_max + 1)

# ---------------------------
# Question 11
# ---------------------------

smooth_poisson = poisson(data_mean, x)
smooth_gaussian = gaussian(data_mean, data_std, x)

# ---------------------------
# Question 12
# Unscaled plot
# ---------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    data,
    bins=np.arange(data_min, data_max + 2),
    alpha=0.6,
    label="Data"
)

plt.plot(
    x,
    smooth_poisson,
    "r-o",
    label="Poisson"
)

plt.plot(
    x,
    smooth_gaussian,
    "g-o",
    label="Gaussian"
)

plt.xlabel("Counts")
plt.ylabel("Frequency")
plt.title("Unscaled Distributions")
plt.legend()

plt.show()

# ---------------------------
# Question 13
# Scale distributions
# ---------------------------

bins = np.arange(data_min, data_max + 2)
bin_width = 1

poisson_scaled = smooth_poisson * len(data) * bin_width
gaussian_scaled = smooth_gaussian * len(data) * bin_width

plt.figure(figsize=(8, 5))

plt.hist(
    data,
    bins=bins,
    alpha=0.6,
    label="Data"
)

plt.plot(
    x,
    poisson_scaled,
    "r-o",
    linewidth=2,
    label="Scaled Poisson"
)

plt.plot(
    x,
    gaussian_scaled,
    "g-o",
    linewidth=2,
    label="Scaled Gaussian"
)

plt.xlabel("Counts")
plt.ylabel("Frequency")
plt.title("Histogram with Scaled Distributions")
plt.legend


# In[ ]:




