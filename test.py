from diffuse_neutrino_flux import Spectrum
import matplotlib.pyplot as plt


def test_plot_spectrum():
    fig, ax = plt.subplots()
    for i, k in enumerate(Spectrum.list_available_spectra()):
        s = Spectrum.from_key(k)
        c = f"C{i}"
        s.plot(ax, log=True, energy_scaling=2, label=s.journal, color=c)
        for cl in s.contour_files:
            s.plot_cl(cl, ax, log=True, energy_scaling=2, alpha=cl / 100, color=c)
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Flux")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("test_plot_spectrum.pdf")
    plt.close()
