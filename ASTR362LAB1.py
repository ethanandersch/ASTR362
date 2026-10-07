#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from glob import glob


# In[4]:


def correct_encoding(image):
    return image / 16 + 1


# In[5]:


base_folder = "/mnt/c/Users/Ethan/OneDrive/Apps/Designer"

obs_folder = f"{base_folder}/OneDrive_1_9-28-2026/observations"

twilight_folder = f"{base_folder}/Twilight"


# In[6]:


bias_files = sorted(
    glob(f"{base_folder}/bias-*.fit")
)

print("Number of bias files:", len(bias_files))
print(bias_files[:5])


# In[7]:


all_bias = []

for file in bias_files:

    data = fits.getdata(file)

    data = correct_encoding(data)

    all_bias.append(data)

all_bias = np.array(all_bias)

print(all_bias.shape)


# In[8]:


master_bias = np.mean(all_bias, axis=0)

print(master_bias.shape)


# In[9]:


median_bias = np.median(master_bias)

print("Median Bias =", median_bias, "ADU")


# In[11]:


plt.figure(figsize=(10,8))

plt.imshow(
    master_bias,
    cmap='viridis',
    origin='lower'
)

plt.colorbar(label='ADU')

plt.title("Master Bias")

plt.savefig(
    "master_bias.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# In[12]:


difference_image = all_bias[0] - all_bias[1]

print(
    "Mean difference image =",
    np.mean(difference_image)
)


# In[13]:


plt.figure(figsize=(8,6))

plt.hist(
    difference_image.ravel(),
    bins=300
)

plt.xlabel("ADU")

plt.ylabel("Pixels")

plt.title("Difference Image Histogram")

plt.show()


# In[15]:


sigma_diff = np.std(difference_image)

sigma_read = sigma_diff / np.sqrt(2)

print("Sigma difference =", sigma_diff)
print("Read noise =", sigma_read, "ADU")


# In[16]:


all_darks = sorted(
    glob(f"{obs_folder}/darks*.fit")
)

print("Number of darks =", len(all_darks))

for f in all_darks[:20]:
    print(f.split("/")[-1])


# In[17]:


dark10_files = sorted(
    glob(f"{obs_folder}/darks_10s*.fit")
)

print("10-second darks:", len(dark10_files))


# In[18]:


dark10_stack = []

for file in dark10_files:

    data = fits.getdata(file)

    data = correct_encoding(data)

    dark10_stack.append(data)

dark10_stack = np.array(dark10_stack)

master_dark10 = np.mean(
    dark10_stack,
    axis=0
)

print(master_dark10.shape)


# In[19]:


master_dark10_bc = master_dark10 - master_bias


# In[20]:


plt.figure(figsize=(10,8))

plt.imshow(
    master_dark10_bc,
    cmap='viridis',
    origin='lower'
)

plt.colorbar(label='ADU')

plt.title("Bias Corrected 10s Dark")

plt.show()


# In[21]:


flat_folder = "/mnt/c/Users/Ethan/OneDrive/Apps/Designer/EthanandGabe"

flat_files = sorted(
    glob(f"{flat_folder}/Flats_Ethan&Gabe-*.fit")
)

print("Number of flats =", len(flat_files))

for f in flat_files[:10]:
    print(f.split("/")[-1])


# In[22]:


flat_subset = flat_files[:10]

print("Using", len(flat_subset), "flats")


# In[23]:


flat_stack = []

for file in flat_subset:

    data = fits.getdata(file)

    data = correct_encoding(data)

    flat_stack.append(data)

flat_stack = np.array(flat_stack, dtype=np.float32)

print(flat_stack.shape)


# In[24]:


normalized_flats = []

for flat in flat_stack:

    flat_bc = flat - master_bias

    median_value = np.median(flat_bc)

    norm_flat = flat_bc / median_value

    normalized_flats.append(norm_flat)

normalized_flats = np.array(
    normalized_flats,
    dtype=np.float32)


# In[25]:


normalized_flats = []

for flat in flat_stack:

    flat_bc = flat - master_bias

    median_value = np.median(flat_bc)

    norm_flat = flat_bc / median_value

    normalized_flats.append(norm_flat)

normalized_flats = np.array(normalized_flats)

print(normalized_flats.shape)


# In[26]:


master_flat = np.median(
    normalized_flats,
    axis=0
)

print(master_flat.shape)


# In[27]:


plt.figure(figsize=(10,8))

plt.imshow(
    master_flat,
    origin='lower',
    cmap='viridis',
    vmin=0.95,
    vmax=1.05
)

plt.colorbar(label='Normalized Response')

plt.title("Median Normalized Flat")

plt.savefig(
    "median_normalized_flat.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# In[28]:


plt.figure(figsize=(8,8))

plt.imshow(
    master_flat[1000:1200,1000:1200],
    origin='lower',
    cmap='viridis'
)

plt.colorbar(label='Normalized Response')

plt.title("Flat Field Detail")

plt.show()


# In[29]:


print("Mean =", np.mean(master_flat))
print("Median =", np.median(master_flat))
print("Standard Deviation =", np.std(master_flat))


# In[30]:


arcturus_files = sorted(
    glob(f"{obs_folder}/arcturus_10sec_*.fit")
)

print(len(arcturus_files))


# In[31]:


science = fits.getdata(arcturus_files[0])

science = correct_encoding(science)

print(science.shape)


# In[32]:


arcturus_reduced = (
    science
    - master_dark10_bc
    - master_bias
) / master_flat


# In[33]:


vmin = np.percentile(arcturus_reduced, 5)
vmax = np.percentile(arcturus_reduced, 99)

plt.figure(figsize=(10,8))

plt.imshow(
    arcturus_reduced,
    origin='lower',
    cmap='gray',
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(label='ADU')

plt.title("Reduced Arcturus")

plt.show()


# In[36]:


plt.figure(figsize=(10,8))

plt.imshow(
    arcturus_reduced,
    origin='lower',
    cmap='gray',
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(label='ADU')

plt.title("Reduced Arcturus")

plt.savefig(
    "Reduced_Arcturus.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# In[38]:


dubhe_files = sorted(
    glob(f"{obs_folder}/dubhe_10sec_*.fit")
)

dubhe = fits.getdata(dubhe_files[0])

dubhe = correct_encoding(dubhe)

dubhe_reduced = (
    dubhe
    - master_dark10_bc
    - master_bias
) / master_flat


# In[40]:


vmin = np.percentile(
    dubhe_reduced,
    5
)

vmax = np.percentile(
    dubhe_reduced,
    99
)

plt.figure(figsize=(10,8))

plt.imshow(
    dubhe_reduced,
    origin='lower',
    cmap='gray',
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(label='ADU')

plt.title("Reduced Dubhe")

plt.show()


# In[6]:


dark30_files = sorted(
    glob(f"{obs_folder}/darks_30s*.fit")
)

print("Number of 30s darks =", len(dark30_files))


# In[7]:


dark30_stack = []

for file in dark30_files:

    data = fits.getdata(file)

    data = correct_encoding(data)

    dark30_stack.append(data)


# In[8]:


master_dark30 = np.mean(
    dark30_stack,
    axis=0
)

print(master_dark30.shape)


# In[18]:


master_dark30_bc = (
    master_dark30
    - master_bias
)


# In[19]:


sadalsuud_files = sorted(
    glob(f"{obs_folder}/sadalsuud_30sec_*.fit")
)

print(len(sadalsuud_files))


# In[20]:


sadalsuud = fits.getdata(
    sadalsuud_files[0]
)

sadalsuud = correct_encoding(
    sadalsuud
)


# In[31]:


sadalsuud_reduced = (
    sadalsuud
    - master_dark30_bc
    - master_bias
) / master_flat


# In[32]:


vmin = np.percentile(
    sadalsuud_reduced,
    5
)

vmax = np.percentile(
    sadalsuud_reduced,
    99
)

plt.figure(figsize=(10,8))

plt.imshow(
    sadalsuud_reduced,
    origin='lower',
    cmap='gray',
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(label='ADU')

plt.title("Reduced Sadalsuud")

plt.show()


# In[ ]:




