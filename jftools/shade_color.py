# taken from https://github.com/matplotlib/matplotlib/pull/2745
# Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
def shade_color(color, percent):
    """
    A color helper utility to either darken or lighten given color.
    This color utility function allows the user to easily darken or lighten a color for
    plotting purposes.  This function first converts the given color to RGB using 
    ColorConverter and then to HSL.  The saturation is modified according to the given 
    percentage and converted back to RGB.
    Parameters
    ----------
    color : string, list, hexvalue
        Any acceptable Matplotlib color value, such as 'red', 'slategrey', '#FFEE11', (1,0,0)
    percent :  the amount by which to brighten or darken the color.
    Returns
    -------
    color : tuple of floats
        tuple representing converted rgb values
    """
    from matplotlib.colors import colorConverter
    from colorsys import rgb_to_hls, hls_to_rgb
    import numpy as np
    r,g,b = colorConverter.to_rgb(color)
    h,l,s = rgb_to_hls(r,g,b)
    l *= 1 + float(percent)/100
    l = np.clip(l, 0, 1)
    r,g,b = hls_to_rgb(h,l,s)
    return r,g,b
