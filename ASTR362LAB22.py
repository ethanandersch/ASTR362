#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits


# In[3]:


data_dir = Path(
    "/mnt/c/Users/Ethan/Downloads/ASTR362LAB2/DATA/Lab 2"
)


# In[4]:


def load_fits(filename):

    data = fits.getdata(filename)

    header = fits.getheader(filename)

    return data, header


# In[5]:


bias_files = sorted(
    data_dir.glob("bias*.fit")
)

flat_files = sorted(
    data_dir.glob("flat*.fit")
)

bias_stack = [
    fits.getdata(f).astype(float)
    for f in bias_files
]

master_bias = np.mean(
    bias_stack,
    axis=0
)


def build_master_dark(pattern):

    dark_files = sorted(
        data_dir.glob(pattern)
    )

    dark_stack = [
        fits.getdata(f).astype(float)
        for f in dark_files
    ]

    return (
        np.mean(dark_stack, axis=0)
        -
        master_bias
    )


master_dark_01 = build_master_dark(
    "darks0.1*.fit"
)

master_dark_5 = build_master_dark(
    "darks5*.fit"
)

master_dark_10 = build_master_dark(
    "darks10*.fit"
)

master_dark_20 = build_master_dark(
    "darks20*.fit"
)

flat_stack = [
    fits.getdata(f).astype(float)
    for f in flat_files[:5]
]

normalized_flats = []

for flat in flat_stack:

    flat_bc = (
        flat
        -
        master_bias
    )

    normalized_flats.append(
        flat_bc /
        np.median(flat_bc)
    )

master_flat = np.median(
    normalized_flats,
    axis=0
)

print("Calibration frames complete")


# In[6]:


def reduce_image(
    image,
    master_bias,
    master_dark,
    master_flat
):

    reduced = (
        image
        -
        master_bias
        -
        master_dark
    )

    reduced = (
        reduced
        /
        master_flat
    )

    return reduced


# In[7]:


def aperture_photometry(
    image,
    x0,
    y0,
    radius=20,
    sky_inner=30,
    sky_outer=50
):

    y, x = np.indices(
        image.shape
    )

    r = np.sqrt(
        (x-x0)**2
        +
        (y-y0)**2
    )

    star_mask = (
        r <= radius
    )

    sky_mask = (
        (r >= sky_inner)
        &
        (r <= sky_outer)
    )

    star_counts = np.sum(
        image[star_mask]
    )

    sky_per_pixel = np.median(
        image[sky_mask]
    )

    sky_counts = (
        sky_per_pixel
        *
        np.sum(star_mask)
    )

    net_counts = (
        star_counts
        -
        sky_counts
    )

    return net_counts


# In[8]:


def measure_star(
    filename,
    dark_frame,
    xcen,
    ycen
):

    image, header = load_fits(
        str(data_dir / filename)
    )

    reduced = reduce_image(
        image,
        master_bias,
        dark_frame,
        master_flat
    )

    counts = aperture_photometry(
        reduced,
        xcen,
        ycen
    )

    exptime = header["EXPTIME"]

    m_inst = (
        -2.5
        *
        np.log10(
            counts/exptime
        )
    )

    return {
        "object": header["OBJECT"],
        "counts": counts,
        "m_inst": m_inst
    }


# In[9]:


coords = {

    "18del":
        (
            "18del-0001.fit",
            master_dark_5,
            1607,
            2470
        ),

    "HD 195088":
        (
            "8magstar-0001.fit",
            master_dark_10,
            3505,
            342
        ),

    "thuban":
        (
            "thuban-0001.fit",
            master_dark_01,
            3307,
            818
        ),

    "mysterystar":
        (
            "mysterystar-0001.fit",
            master_dark_20,
            950,
            2150
        ),

    "3land":
        (
            "3land-0001.fit",
            master_dark_5,
            2083,
            1758
        ),

    "274_820":
        (
            "2xlan_820-0001.fit",
            master_dark_5,
            2080,
            2000
        ),

    "276_820":
        (
            "2xlan_820-0001.fit",
            master_dark_5,
            1980,
            1880
        ),

    "274_94":
        (
            "94lan2x-0001.fit",
            master_dark_10,
            3800,
            2250
        ),

    "276_94":
        (
            "94lan2x-0001.fit",
            master_dark_10,
            3750,
            2125
        )

}


# In[10]:


results = {}

for key in coords:

    file, dark, x, y = coords[key]

    results[key] = measure_star(
        file,
        dark,
        x,
        y
    )

for key in results:

    print(
        key,
        results[key]["m_inst"]
    )


# In[11]:


known_mags = {

    "SA113-267": 7.653,

    "SA113-274": 8.831,

    "SA113-276": 9.074

}


# In[12]:


C_values = np.array([

    known_mags["SA113-267"]
    - results["3land"]["m_inst"],

    known_mags["SA113-274"]
    - results["274_820"]["m_inst"],

    known_mags["SA113-276"]
    - results["276_820"]["m_inst"],

    known_mags["SA113-274"]
    - results["274_94"]["m_inst"],

    known_mags["SA113-276"]
    - results["276_94"]["m_inst"]

])

print(C_values)


# In[13]:


airmass = np.array([

    1.46,   # SA113-267

    1.46,   # SA113-274 @ 8:20

    1.46,   # SA113-276 @ 8:20

    1.62,   # SA113-274 @ 9:40

    1.62    # SA113-276 @ 9:40

])


# In[14]:


coeffs = np.polyfit(
    airmass,
    C_values,
    1
)

k = coeffs[0]
ZP = coeffs[1]

print("k =", k)
print("ZP =", ZP)


# In[15]:


plt.figure(figsize=(8,6))

plt.scatter(
    airmass,
    C_values,
    s=75
)

xfit = np.linspace(
    np.min(airmass),
    np.max(airmass),
    100
)

plt.plot(
    xfit,
    k*xfit + ZP,
    color="red"
)

plt.xlabel("Airmass (sec z)")
plt.ylabel("C")

plt.title(
    "Landolt Calibration"
)

plt.grid()

plt.show()


# In[16]:


science_airmass = {

    "18del": 1.46,

    "HD 195088": 1.1,

    "thuban": 2.01,

    "mysterystar": 1.50

}


# In[17]:


for star in [
    "18del",
    "HD 195088",
    "thuban",
    "mysterystar"
]:

    m_final = (
        results[star]["m_inst"]
        +
        ZP
        +
        k*science_airmass[star]
    )

    print()

    print(star)

    print(
        "Final Magnitude:",
        round(m_final,3)
    )


# In[19]:


science_stars = [
    ("18del-0001.fit", master_dark_5),
    ("8magstar-0001.fit", master_dark_10),
    ("thuban-0001.fit", master_dark_01),
    ("mysterystar-0001.fit", master_dark_20)
]

for file, dark in science_stars:

    image, header = load_fits(
        str(data_dir / file)
    )

    reduced = reduce_image(
        image,
        master_bias,
        dark,
        master_flat
    )

    print(
        file,
        np.max(reduced)
    )


# In[20]:


science_stars = {
    "18del":
        ("18del-0001.fit", master_dark_5),
    "8magstar":
        ("8magstar-0001.fit", master_dark_10),
    "thuban":
        ("thuban-0001.fit", master_dark_01),
    "mysterystar":
        ("mysterystar-0001.fit", master_dark_20)
}

for star in science_stars:

    file, dark = science_stars[star]

    image, header = load_fits(
        str(data_dir / file)
    )

    reduced = reduce_image(
        image,
        master_bias,
        dark,
        master_flat
    )

    print(
        star,
        np.max(reduced)
    )


# In[21]:


detector_limit = 65535

for star in science_stars:

    file, dark = science_stars[star]

    image, header = load_fits(
        str(data_dir / file)
    )

    reduced = reduce_image(
        image,
        master_bias,
        dark,
        master_flat
    )

    peak = np.max(reduced)

    if peak > 0.95 * detector_limit:

        print(
            star,
            "POTENTIALLY SATURATED",
            peak
        )

    else:

        print(
            star,
            "OK",
            peak
        )


# In[22]:


final_table = {}

for star in [
    "18del",
    "HD 195088",
    "thuban",
    "mysterystar"
]:
    ZP_std = np.std(C_values)
    mag = (
        results[star]["m_inst"]
        +
        ZP
        +
        k*science_airmass[star]
    )

    err = np.sqrt(
        ZP_std**2
    )

    final_table[star] = {
        "mag": mag,
        "err": err
    }

    print(
        f"{star:12s}"
        f"{mag:10.3f}"
        f"{err:10.3f}"
    )


# In[23]:


science_stars = {

    "18 Del":
        ("18del-0001.fit",
         master_dark_5),

    "HD 195088":
        ("8magstar-0001.fit",
         master_dark_10),

    "Thuban":
        ("thuban-0001.fit",
         master_dark_01),

    "Mystery Star":
        ("mysterystar-0001.fit",
         master_dark_20)

}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,12)
)

axes = axes.flatten()

for i, star in enumerate(science_stars):

    file, dark = science_stars[star]

    image, header = load_fits(
        str(data_dir / file)
    )

    reduced = reduce_image(
        image,
        master_bias,
        dark,
        master_flat
    )

    axes[i].imshow(
        reduced,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(reduced,5),
        vmax=np.percentile(reduced,99)
    )

    axes[i].set_title(star)

plt.tight_layout()

plt.savefig(
    "science_star_fields.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[24]:


x = 1607
y = 2470

radius = 20

image, header = load_fits(
    str(data_dir / "18del-0001.fit")
)

reduced = reduce_image(
    image,
    master_bias,
    master_dark_5,
    master_flat
)

fig, ax = plt.subplots(
    figsize=(8,8)
)

ax.imshow(
    reduced,
    origin="lower",
    cmap="gray",
    vmin=np.percentile(reduced,50),
    vmax=np.percentile(reduced,99.8)
)

circle = plt.Circle(
    (x,y),
    radius,
    fill=False,
    color="red",
    linewidth=2
)

ax.add_patch(circle)

ax.set_xlim(
    x-100,
    x+100
)

ax.set_ylim(
    y-100,
    y+100
)

ax.set_title(
    "18 Del Aperture Photometry"
)

plt.savefig(
    "18del_aperture.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[25]:


image, header = load_fits(
    str(data_dir / "thuban-0001.fit")
)

reduced = reduce_image(
    image,
    master_bias,
    master_dark_01,
    master_flat
)

radii = np.arange(
    5,
    51,
    5
)

counts = []

for r in radii:

    counts.append(
        aperture_photometry(
            reduced,
            3307,
            818,
            radius=r,
            sky_inner=r+10,
            sky_outer=r+30
        )
    )

plt.figure(figsize=(8,6))

plt.plot(
    radii,
    counts,
    marker="o"
)

plt.xlabel(
    "Aperture Radius (pixels)"
)

plt.ylabel(
    "Net Counts"
)

plt.title(
    "Growth Curve"
)

plt.grid()

plt.savefig(
    "growth_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[26]:


plt.figure(figsize=(8,6))

plt.scatter(
    airmass,
    C_values,
    s=80
)

xfit = np.linspace(
    np.min(airmass),
    np.max(airmass),
    100
)

plt.plot(
    xfit,
    k*xfit + ZP,
    color="red",
    linewidth=2
)

plt.xlabel(
    "Airmass (sec z)"
)

plt.ylabel(
    "Calibration Constant C"
)

plt.title(
    "Landolt Calibration"
)

plt.grid()

plt.savefig(
    "landolt_calibration.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[27]:


science_stars = {

    "18del":
        ("18del-0001.fit",
         master_dark_5),

    "HD 195088":
        ("8magstar-0001.fit",
         master_dark_10),

    "thuban":
        ("thuban-0001.fit",
         master_dark_01),

    "mysterystar":
        ("mysterystar-0001.fit",
         master_dark_20)

}

detector_limit = 65535

print(
    f"{'Star':15s}"
    f"{'Peak ADU':>12s}"
    f"{'Status':>15s}"
)

for star in science_stars:

    file, dark = science_stars[star]

    image, header = load_fits(
        str(data_dir / file)
    )

    reduced = reduce_image(
        image,
        master_bias,
        dark,
        master_flat
    )

    peak = np.max(reduced)

    status = (
        "SATURATED"
        if peak > detector_limit
        else "OK"
    )

    print(
        f"{star:15s}"
        f"{peak:12.0f}"
        f"{status:>15s}"
    )


# In[28]:


image, header = load_fits(
    str(data_dir / "thuban-0001.fit")
)

reduced = reduce_image(
    image,
    master_bias,
    master_dark_01,
    master_flat
)

radii = np.arange(5,51,5)

counts = []
pixels = []

for r in radii:

    flux = aperture_photometry(
        reduced,
        3307,
        818,
        radius=r,
        sky_inner=r+10,
        sky_outer=r+30
    )

    counts.append(flux)

    pixels.append(np.pi*r**2)

plt.figure(figsize=(8,6))

plt.plot(
    pixels,
    counts,
    marker="o"
)

plt.xlabel(
    "Pixels in Aperture"
)

plt.ylabel(
    "Net Counts"
)

plt.title(
    "Growth Curve"
)

plt.grid()

plt.savefig(
    "growth_curve_pixels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[29]:


coeffs, cov = np.polyfit(
    airmass,
    C_values,
    1,
    cov=True
)

k = coeffs[0]
ZP = coeffs[1]

k_err = np.sqrt(cov[0,0])
ZP_err = np.sqrt(cov[1,1])

print("k =", k, "+/-", k_err)
print("ZP =", ZP, "+/-", ZP_err)


# In[30]:


flat_files = sorted(
    data_dir.glob("flat*.fit")
)

signal = []
variance = []

for i in range(0, min(len(flat_files)-1,20), 2):

    flat1 = fits.getdata(
        flat_files[i]
    ).astype(float)

    flat2 = fits.getdata(
        flat_files[i+1]
    ).astype(float)

    flat1 = flat1 - master_bias
    flat2 = flat2 - master_bias

    avg = (
        flat1 + flat2
    ) / 2

    diff = (
        flat1 - flat2
    )

    signal.append(
        np.mean(avg)
    )

    variance.append(
        np.var(diff) / 2
    )

signal = np.array(signal)
variance = np.array(variance)

coeffs, cov = np.polyfit(
    signal,
    variance,
    1,
    cov=True
)

slope = coeffs[0]
intercept = coeffs[1]

slope_err = np.sqrt(
    cov[0,0]
)

gain = 1/slope

gain_err = slope_err/(slope**2)

xfit = np.linspace(
    signal.min(),
    signal.max(),
    200
)

plt.figure(figsize=(8,6))

plt.scatter(
    signal,
    variance,
    s=50
)

plt.plot(
    xfit,
    slope*xfit + intercept,
    color="red"
)

plt.xlabel(
    "Signal Level (ADU)"
)

plt.ylabel(
    "Variance (ADU$^2$)"
)

plt.title(
    "Photon Transfer Gain Curve"
)

plt.grid()

plt.savefig(
    "gain_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Gain = {gain:.3f} +/- {gain_err:.3f} e-/ADU"
)


# In[31]:


image, header = load_fits(
    str(data_dir / "18del-0001.fit")
)

reduced = reduce_image(
    image,
    master_bias,
    master_dark_5,
    master_flat
)

x0 = 1607
y0 = 2470

y, x = np.indices(
    reduced.shape
)

r = np.sqrt(
    (x-x0)**2 +
    (y-y0)**2
)

r_int = r.astype(int)

tbin = np.bincount(
    r_int.ravel(),
    reduced.ravel()
)

nr = np.bincount(
    r_int.ravel()
)

radial_profile = tbin / nr

plt.figure(figsize=(8,6))

plt.plot(
    radial_profile[:100]
)

plt.axvline(
    20,
    color="red",
    linestyle="--",
    label="Chosen Aperture"
)

plt.xlabel(
    "Radius (pixels)"
)

plt.ylabel(
    "Mean Counts"
)

plt.title(
    "Radial Brightness Profile"
)

plt.legend()

plt.grid()

plt.savefig(
    "radial_profile.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[33]:


star = "HD 195088"

counts = results[star]["counts"]

radius = 20

n_pixels = np.pi * radius**2

read_noise = 4.26

# photon-noise estimate
snr = counts / np.sqrt(
    counts +
    n_pixels*(read_noise**2)
)

sigma_inst = 1/snr

sigma_cal = np.std(C_values)

sigma_total = np.sqrt(
    sigma_inst**2 +
    sigma_cal**2
)

print("Star:", star)
print()

print("Net Counts =", counts)

print("Pixels in Aperture =", n_pixels)

print("SNR =", snr)

print(
    "Instrumental Uncertainty =",
    sigma_inst
)

print(
    "Calibration Uncertainty =",
    sigma_cal
)

print(
    "Total Uncertainty =",
    sigma_total
)


# In[35]:


fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,12)
)

plots = [

    ("18 Del",
     "18del-0001.fit",
     master_dark_5,
     1607,
     2470),

    ("HD 195088",
     "8magstar-0001.fit",
     master_dark_10,
     3505,
     342),

    ("Thuban",
     "thuban-0001.fit",
     master_dark_01,
     3307,
     818),

    ("Mystery Star",
     "mysterystar-0001.fit",
     master_dark_20,
     950,
     2150)

]

for ax, (name, file, dark, x, y) in zip(
    axes.flatten(),
    plots
):

    image, header = load_fits(
        str(data_dir / file)
    )

    reduced = reduce_image(
        image,
        master_bias,
        dark,
        master_flat
    )

    ax.imshow(
        reduced,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(reduced,5),
        vmax=np.percentile(reduced,99)
    )

    circle = plt.Circle(
        (x, y),
        50,
        fill=False,
        color="red",
        linewidth=2
    )

    ax.add_patch(circle)

    ax.scatter(
        x,
        y,
        color="red",
        s=30
    )

    ax.set_title(name)

plt.tight_layout()

plt.savefig(
    "science_star_fields_marked.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:




