[![CI](https://github.com/JannisNe/diffuse_neutrino_flux/actions/workflows/test.yml/badge.svg)](https://github.com/JannisNe/timewise/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/diffuse_neutrino_flux.svg)](https://badge.fury.io/py/diffuse_neutrino_flux)
# Add diffuse neutrino flux measurements to your plots

## Usage

The central object is the `Spectrum`. Curently implented spectral shapes are the `SinglePowerLaw`, `BrokenPowerLaw` and the `LogParabola`. The impleneted measurements can be found in `diffuse_neutrino_flux/data/measurements.json`. They can be shown by 
```python
from diffuse_neutrino_flux import Spectrum
print(Spectrum.list_available_spectra())
```

The spectrum can be plotted. If a contour file is available, also the corresponding butterfly can be shown. The `energy_scaling` parameter can be used to scale the energy axis. All other parameters are passed to the `matplotlib` plot function. 

```python
import matplotlib.pyplot as plt
from diffuse_neutrino_flux import Spectrum

s = Spectrum.from_key("joint15")

fig, ax = plt.subplots()
s.plot(ax=ax, label="Joint15", color="blue", energy_scaling=2)
s.plot_cl(ax=ax, color="blue", alpha=0.5, energy_scaling=2)
ax.set_xlabel("Energy [GeV]")
ax.set_ylabel("Flux [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
plt.show()
```

![Joint15 spectrum](example.png)

## Adding new measurements
To add a new measurement, you need to create a new entry in `diffuse_neutrino_flux/data/measurements.json`. To be able to plot the butterfly, you also need to create a contour file. Simply trace the contour scan in the parameter plane and add the data as a CSV file in `diffuse_neutrino_flux/data/` as `<new_measurement_name>_contour<confidence_level>.csv`. 