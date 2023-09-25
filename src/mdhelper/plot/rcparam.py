"""
Matplotlib rcParams
===================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module provides optimized Matplotlib rcParams for various
scientific publications.
"""

import matplotlib as mpl

# Define figure size guidelines for various publications in inches
MAX_FIGURE_SIZES = {
    "acs": {
        "width": 3.25,
        "double_width": 7,
        "length": 9.5
    },
    "rsc": {
        "width": 3.26771654,
        "double_width": 6.73228346,
        "length": 9.17322835
    }
}

def update(journal: str = None, **kwargs) -> None:

    """
    Updates the Matplotlib rcParams at runtime. By default, this
    function overwrites the following settings:

    * :code:`axes.labelsize`: :code:`14`
    * :code:`figure.autolayout`: :code:`True`
    * :code:`font.size`: :code:`12`
    * :code:`grid.color`: :code:`0`
    * :code:`grid.linestyle`: :code:`(0, (1, 5))`
    * :code:`legend.columnspacing`: :code:`1`
    * :code:`legend.edgecolor`: :code:`1`
    * :code:`legend.fontsize`: :code:`12`
    * :code:`legend.handlelength`: :code:`1.25`
    * :code:`legend.labelspacing`: :code:`0.25`
    * :code:`savefig.dpi`: :code:`1200`
    * :code:`xtick.labelsize`: :code:`12`
    * :code:`ytick.labelsize`: :code:`12`
    * :code:`text.usetex`: :code:`True`

    If a supported journal acronym is provided as the first argument,
    the default figure size will also be updated.

    Parameters
    ----------
    journal : `str`, optional
        Journal acronym used to update the default figure size.

        .. container::    
    
           **Valid values**:
        
           * :code:`acs`: American Chemical Society.
           * :code:`rsc`: Royal Society of Chemistry.
    
    **kwargs : ...
        Additional rcParams to update passed to
        :meth:`matplotlib.rcParams.update`.
    """
    
    fig_size = {} if journal is None else {
        "figure.figsize": (
            1.5 * MAX_FIGURE_SIZES[journal]["width"],
            1.125 * MAX_FIGURE_SIZES[journal]["width"]
        )
    }
    
    mpl.rcParams.update(
        {
            "axes.labelsize": 14,
            "figure.autolayout": True,
            "font.size": 12,
            "grid.color": "0",
            "grid.linestyle": (0, (1, 5)),
            "legend.columnspacing": 1,
            "legend.edgecolor": "1",
            "legend.fontsize": 12,
            "legend.handlelength": 1.25,
            "legend.labelspacing": 0.25,
            "savefig.dpi": 1200,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "text.usetex": True
        } | fig_size | kwargs
    )