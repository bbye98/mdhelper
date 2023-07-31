"""
Utility functions
=================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains a collection of utility functions used by other
MDHelper modules.
"""

from datetime import datetime

def log(text: str) -> None:

    """
    Log information to console with the datetime prefixed.

    Parameters
    ----------
    text : `str`
        Message to log.
    """

    text = text.replace("\t", 4 * " ").replace("\n", f"\n{10 * ' '}")
    print(f"{datetime.now().strftime('%H:%M:%S')}  {text}")