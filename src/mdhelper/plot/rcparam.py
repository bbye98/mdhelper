"""
Matplotlib rcParams
===================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module provides optimized Matplotlib rcParams for various
scientific publications.
"""

import matplotlib as mpl

# Define figure size guidelines for various publications in inches
FIGURE_SIZE_LIMITS = {
    "acs": {"max_single_width": 3.25, "max_double_width": 7, 
            "max_length": 9.5},
    "aip": {"max_single_width": 3.37, "max_double_width": 6.69, 
            "max_length": 8.25, "min_font_size": 8},
    "rsc": {"max_single_width": 3.26771654, "max_double_width": 6.73228346, 
            "max_length": 9.17322835}
}

def update(journal: str = None, **kwargs) -> None:

    """
    Updates the Matplotlib rcParams at runtime. By default, this
    function overwrites the following settings:

    .. code::

       {
           "axes.labelsize": 9,
           "figure.autolayout": True,
           "font.size": 9,
           "legend.columnspacing": 1,
           "legend.edgecolor": "1",
           "legend.fontsize": 9,
           "legend.handlelength": 1.25,
           "legend.labelspacing": 0.25,
           "savefig.dpi": 1_200,
           "xtick.labelsize": 9,
           "ytick.labelsize": 9,
           "text.usetex": True
       }

    If a supported journal acronym is provided as the first argument,
    the default figure size will also be updated.

    Parameters
    ----------
    journal : `str`, optional
        Journal acronym used to update the default figure size.

        .. container::    
    
           **Valid values**:
        
           * :code:`"acs"`: American Chemical Society.
           * :code:`"aip"`: American Institute of Physics.
           * :code:`"rsc"`: Royal Society of Chemistry.
    
    **kwargs
        Additional rcParams to update passed to
        :meth:`matplotlib.rcParams.update`.
    """
    
    fig_size = {} if journal is None else {
        "figure.figsize": (FIGURE_SIZE_LIMITS[journal]["max_single_width"],
                           3 * FIGURE_SIZE_LIMITS[journal]["max_single_width"] / 4)
    }
    
    mpl.rcParams.update(
        {
            "axes.labelsize": 9,
            "figure.autolayout": True,
            "font.size": 9,
            "legend.columnspacing": 1,
            "legend.edgecolor": "1",
            "legend.fontsize": 9,
            "legend.handlelength": 1.25,
            "legend.labelspacing": 0.25,
            "savefig.dpi": 1_200,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.usetex": True
        } | fig_size | kwargs
    )