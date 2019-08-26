from nwbwidgets import view
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from scipy.ndimage.filters import gaussian_filter

# to make readable plot, show first mx_pts points or less
mx_pts = 4000

def show_position(node, neurodata_vis_spec):

    if len(node.spatial_series.keys()) == 1:
        for value in node.spatial_series.values():
            return view.nwb2widget(value, neurodata_vis_spec=neurodata_vis_spec)
    else:
        return view.nwb2widget(node.spatial_series, neurodata_vis_spec=neurodata_vis_spec)


def show_spatial_series(node, **kwargs):

    text_wiget = view.show_text_fields(node, exclude=('timestamps_unit', 'comments'))

    if node.conversion and np.isfinite(node.conversion):
        data = node.data * node.conversion
        unit = node.unit
    else:
        data = node.data
        unit = None

    fig, ax = plt.subplots()
    sp_len = len(data)
    if data.shape[0] > 1:
        if node.timestamps:
            ax.plot(node.timestamps[0:np.minimum(sp_len,mx_pts)], data[0:np.minimum(sp_len,mx_pts)])
        else:
            ax.plot(np.arange(np.minimum(sp_len,mx_pts)) / node.rate, data[0:np.minimum(sp_len,mx_pts)])
        ax.set_xlabel('t (sec)')

        if unit:
            ax.set_ylabel('x ({})'.format(unit))
        else:
            ax.set_ylabel('x')
    elif data.shape[0] == 1:
        if node.timestamps:
            ax.plot(node.timestamps[0:np.minimum(sp_len,mx_pts)], data[0:np.minimum(sp_len,mx_pts)])
        else:
            ax.plot(np.arange(np.minimum(sp_len,mx_pts)) / node.rate, data[0:np.minimum(sp_len,mx_pts)])
        ax.set_xlabel('t (sec)')

        if unit:
            ax.set_xlabel('x ({})'.format(unit))
        else:
            ax.set_xlabel('x')
        ax.set_ylabel('x')
    elif data.shape[1] == 2:
        ax.plot(data[:, 0], data[:, 1])
        if unit:
            ax.set_xlabel('x ({})'.format(unit))
            ax.set_ylabel('y ({})'.format(unit))
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        ax.axis('equal')

    return widgets.HBox([text_wiget, view.fig2widget(fig)])

def show_spatial_series_speed(node, **kwargs):

    text_wiget = view.show_text_fields(node, exclude=('timestamps_unit', 'comments'))

    if node.conversion and np.isfinite(node.conversion):
        data = node.data * node.conversion
        unit = node.unit
    else:
        data = node.data
        unit = None
    dx = np.diff(data)
    dt = 1/node.rate
    speed = np.abs(dx/dt)

    # if speed > threshold, set to nan because these values will skew the plot
    mean_speed = np.nanmean(speed)
    std_speed = np.nanstd(speed)
    speed_threshold = mean_speed + 3*std_speed
    idx_outliers = np.where(speed > speed_threshold)
    speed[idx_outliers[0]] = np.nan
    smoothed_speed = gaussian_filter(speed, sigma=3)

    fig, ax = plt.subplots()
    sp_len = len(smoothed_speed)
    if smoothed_speed.shape[0] > 1:
        if node.timestamps:
            ax.plot(node.timestamps[0:np.minimum(sp_len,mx_pts)], smoothed_speed[0:np.minimum(sp_len,mx_pts)])
        else:
            ax.plot(np.arange(np.minimum(sp_len,mx_pts)) / node.rate, smoothed_speed[0:np.minimum(sp_len,mx_pts)])
        ax.set_xlabel('t (sec)')

        if unit:
            ax.set_ylabel('smoothed speed ({}/s)'.format(unit))
        else:
            ax.set_ylabel('smoothed speed')

    elif smoothed_speed.shape[0] == 1:
        if node.timestamps:
            ax.plot(node.timestamps[0:np.minimum(sp_len,mx_pts)], smoothed_speed[0:np.minimum(sp_len,mx_pts)])
        else:
            ax.plot(np.arange(np.minimum(sp_len,mx_pts)) / node.rate, smoothed_speed[0:np.minimum(sp_len,mx_pts)])
        ax.set_xlabel('t (sec)')

        if unit:
            ax.set_xlabel('smoothed speed ({}/s)'.format(unit))
        else:
            ax.set_xlabel('smoothed speed')
        ax.set_ylabel('smoothed speed')
    elif speed.shape[1] == 2:
        ax.plot(speed[:, 0], speed[:, 1])
        if unit:
            ax.set_xlabel('x ({})'.format(unit))
            ax.set_ylabel('y ({})'.format(unit))
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        ax.axis('equal')

    return widgets.HBox([text_wiget, view.fig2widget(fig)])
