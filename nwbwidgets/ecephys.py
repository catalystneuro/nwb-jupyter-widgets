from nwbwidgets import view
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
import itkwidgets
import itk
from scipy.signal import stft


def show_lfp(node, **kwargs):
    lfp = node.electrical_series['lfp']
    ntabs = 3
    children = [widgets.HTML('Rendering...') for _ in range(ntabs)]

    def on_selected_index(change):
        if change.new == 1 and isinstance(change.owner.children[1], widgets.HTML):
            slider = widgets.IntSlider(value=0, min=0, max=lfp.data.shape[1] - 1, description='Channel',
                                       orientation='horizontal')

            def create_spectrogram(channel=0):
                f, t, Zxx = stft(lfp.data[:, channel], lfp.rate, nperseg=128)
                spect = np.log(np.abs(Zxx))
                image = itk.GetImageFromArray(spect)
                image.SetSpacing([(f[1] - f[0]), (t[1] - t[0]) * 1e-1])
                direction = image.GetDirection()
                vnl_matrix = direction.GetVnlMatrix()
                vnl_matrix.set(0, 0, 0.0)
                vnl_matrix.set(0, 1, -1.0)
                vnl_matrix.set(1, 0, 1.0)
                vnl_matrix.set(1, 1, 0.0)
                return image

            spectrogram = create_spectrogram(0)

            viewer = itkwidgets.view(spectrogram, ui_collapsed=True, select_roi=True, annotations=False)
            spect_vbox = widgets.VBox([slider, viewer])
            children[1] = spect_vbox
            change.owner.children = children
            channel_to_spectrogram = {0: spectrogram}

            def on_change_channel(change):
                channel = change.new
                if channel not in channel_to_spectrogram:
                    channel_to_spectrogram[channel] = create_spectrogram(channel)
                viewer.image = channel_to_spectrogram[channel]

            slider.observe(on_change_channel, names='value')

    vbox = []
    for key, value in lfp.fields.items():
        vbox.append(widgets.Text(value=repr(value), description=key, disabled=True))
    children[0] = widgets.VBox(vbox)

    tab_nest = widgets.Tab()
    # Use Rendering... as a placeholder
    tab_nest.children = children
    tab_nest.set_title(0, 'Fields')
    tab_nest.set_title(1, 'Spectrogram')
    tab_nest.set_title(2, 'test')
    tab_nest.observe(on_selected_index, names='selected_index')
    return tab_nest


def show_spectrogram(neurodata, channel=0, **kwargs):
    fig, ax = plt.subplots()
    f, t, Zxx = stft(neurodata.data[:, channel], neurodata.rate, nperseg=2*17)
    ax.imshow(np.log(np.abs(Zxx)), aspect='auto', extent=[0, max(t), 0, max(f)], origin='lower')
    ax.set_ylim(0, 50)
    plt.show(ax.figure())

def show_session_raster(node):
    df = build_table(node)
    #unit_id = 7
    #spike_times = df.iloc[df.index == 7]['spike_times']
    #index_list = df.index.tolist()
    spike_times = df.spike_times[0:100].tolist()

    fig, ax = plt.subplots(1, 1)
    ax.figure.set_size_inches(12,6)
    ax.eventplot(spike_times)
    return fig


def build_table(node):
    df = node.to_dataframe(exclude=set(['timeseries', 'timeseries_index']))
    df.sort_values(by='id', axis=0, inplace=True)
    return df