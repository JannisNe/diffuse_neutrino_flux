from diffuse_neutrino_flux import Spectrum
import matplotlib.pyplot as plt


def test_plot_spectrum():
    fig, ax = plt.subplots()
    for k in Spectrum.list_available_spectra():
        s = Spectrum.from_key(k)
        s.plot(ax, log=True, energy_scaling=2, label=s.journal)
        for cl in s.contour_files:
            s.plot_cl(cl, ax, log=True, energy_scaling=2, alpha=cl / 100)
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Flux")
    ax.legend()
    fig.savefig("test_plot_spectrum.pdf")
    plt.close()
