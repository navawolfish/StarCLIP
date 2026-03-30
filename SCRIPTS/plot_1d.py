import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch
from pypeit import specobjs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def plot_pypeit_spec1d(spec1d_files, output_file=None):
    if not isinstance(spec1d_files, list):
        spec1d_files = [spec1d_files]
    nexp = len(spec1d_files)

    plt.figure(figsize=(100, 10))
    gs = GridSpec(2, nexp, height_ratios=[2, 1])
    gs.update(hspace=0)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])

    for i, spec1d_file in enumerate(spec1d_files):
        try:
            sobjs = specobjs.SpecObjs.from_fitsfile(spec1d_file)
            if len(sobjs) > 1:
                print(f"At least 1 spurious source in {spec1d_file.name}")
        except:
            print(f"Failed to load {spec1d_file.name}")
            continue
        ymax1 = -np.inf
        ymax2 = -np.inf
        for j, sobj in enumerate(sobjs):
            try:
                ax1.plot(
                    sobj.OPT_WAVE[sobj.OPT_MASK],
                    sobj.OPT_COUNTS[sobj.OPT_MASK],
                    c='k',
                    alpha=0.5
                )
                ax1.plot(
                    sobj.OPT_WAVE[sobj.OPT_MASK],
                    sobj.OPT_COUNTS_SKY[sobj.OPT_MASK],
                    c='k',
                    ls='--',
                    alpha=0.5
                )
                ax2.scatter(
                    sobj.OPT_WAVE[sobj.OPT_MASK],
                    (sobj.OPT_COUNTS * np.sqrt(sobj.OPT_COUNTS_IVAR))[sobj.OPT_MASK],
                    marker='.',
                    c='k',
                    alpha=0.5
                )
                ymax1 = np.max([ymax1, 1.5 * np.quantile(sobj.OPT_COUNTS[sobj.OPT_MASK], 0.95)])
                ymax2 = np.max([ymax2, 1.5 * np.quantile((sobj.OPT_COUNTS * np.sqrt(sobj.OPT_COUNTS_IVAR))[sobj.OPT_MASK], 0.95)])
            except TypeError:
                print(f"Failed to load optimal extraction for {spec1d_file.name}, trying box extraction")
                try:
                    ax1.plot(
                        sobj.BOX_WAVE[sobj.BOX_MASK],
                        sobj.BOX_COUNTS[sobj.BOX_MASK],
                        c='k',
                        alpha=0.5
                    )
                    ax1.plot(
                        sobj.BOX_WAVE[sobj.BOX_MASK],
                        sobj.BOX_COUNTS_SKY[sobj.BOX_MASK],
                        c='k',
                        ls='--',
                        alpha=0.5
                    )
                    ax2.scatter(
                        sobj.BOX_WAVE[sobj.BOX_MASK],
                        (sobj.BOX_COUNTS * np.sqrt(sobj.BOX_COUNTS_IVAR))[sobj.BOX_MASK],
                        marker='.',
                        c='k',
                        alpha=0.5
                    )
                    ymax1 = np.max([ymax1, 1.5 * np.quantile(sobj.BOX_COUNTS[sobj.BOX_MASK], 0.95)])
                    ymax2 = np.max([ymax2, 1.5 * np.quantile((sobj.BOX_COUNTS * np.sqrt(sobj.BOX_COUNTS_IVAR))[sobj.BOX_MASK], 0.95)])
                except TypeError:
                    print(f"Failed to load box extraction too")
                    continue
    ax2.set_ylabel('Wavelength [AA]', fontsize=48)
    ax1.set_ylabel('Flux', fontsize=48)
    ax2.set_ylabel('S/N', fontsize=48)
    ax1.tick_params('x', labelsize=0)
    ax2.tick_params('x', labelsize=36)
    ax1.tick_params('y', labelsize=36)
    ax2.tick_params('y', labelsize=36)
    ax1.set_ylim(0, ymax1)
    ax2.set_ylim(0, ymax2)
    if output_file:
        plt.savefig(output_file)
    plt.show()