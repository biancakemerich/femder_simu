# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:58:35 2020

@author: gutoa
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from matplotlib import gridspec
import scipy.signal.windows as win
import scipy.signal as signal
import more_itertools
import femder as fd
import pytta 

def find_nearest(array, value):
    """
    Function to find closest frequency in frequency array.

    Parameters
    ----------
    array : array
        1D array in which to search for the closest value.
    value : float or int
        Value to be searched.

    Returns
    -------
    Closest value found and its position index.
    """
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def closest_node(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    from detect_peaks import detect_peaks
    x = np.random.randn(100)
    x[60:81] = np.nan
    # detect all peaks and plot data
    ind = detect_peaks(x, show=True)
    print(ind)

    x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # set minimum peak height = 0 and minimum peak distance = 20
    detect_peaks(x, mph=0, mpd=20, show=True)

    x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    # set minimum peak distance = 2
    detect_peaks(x, mpd=2, show=True)

    x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # detection of valleys instead of peaks
    detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    x = [0, 1, 1, 0, 1, 1, 0]
    # detect both edges
    detect_peaks(x, edge='both', show=True)

    x = [-2, 1, -2, 2, 1, 1, 3, 0]
    # set threshold = 2
    detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()
        
def SBIR(IR, t_IR, fmin, fmax, winCheck=False, spectraCheck=False, ms=32, method='constant', beta=1, cosWin=False,
         ABEC=False, delta_ABEC=52):
    """

    Function to calculate Speaker Boundary Interference Response

    Parameters
    ----------
    IR, t_IR: 1D arrays, contain bothe Impulse Response magnitude and time step values.
             freq, frf, t, IR = bem(args)

    fmin, fmax: int, minimun and maximum frequency of interest.
            fmin, fmax = 20, 100

    winCheck: bool, option to view windowing in time domain.
            winCheck = True or False

    spectraCheck: bool, option to view frequency response and SBIR in frequency domain.
            spectraCheck = True or False

    modalCheck: bool, option to view room modes prediction of BEM simulation and cuboid approximation.
            modalCheck = True or False
    """

    if len(IR) < 20:
        print('IR resolution not high enough to calculate SBIR')

    if method == 'constant':
        peak = 0  # Window from the start of the IR
        dt = (max(t_IR) / len(t_IR))  # Time axis resolution
        tt_ms = round((ms / 1000) / dt)  # Number of samples equivalent to 64 ms

        # Windows
        post_peak = np.zeros((len(IR[:])))
        pre_peak = np.zeros((len(IR[:])))

        if cosWin is True:
            win_cos = win.cosine(int(2 * tt_ms)) ** 2  # Cosine squared window
        else:
            win_cos = win.tukey(int(2 * tt_ms), beta)  # Cosine window

        window = np.zeros((len(IR[:])))  # Final window
        ##
        win_cos[0:int(tt_ms)] = 1
        window[0:int(2 * tt_ms)] = win_cos
        ##

    elif method == 'peak':
        # Sample of the initial peak
        peak = detect_peaks(IR, mph=(max(IR) * 0.9), threshold=0, edge='rising', show=False)
        if len(peak) > 1:
            peak = peak[0]
            # print('More than one peak at the IR')
        #        ind[x] = 0; # Window max from the beginning
        # peak = 0  # Window from the start of the IR
        dt = (max(t_IR) / len(t_IR))  # Time axis resolution
        tt_ms = round((ms / 1000) / dt)  # Number of samples equivalent to 64 ms

        # Windows
        post_peak = np.zeros((len(IR[:])))
        pre_peak = np.zeros((len(IR[:])))
        win_cos = win.tukey(int(2 * tt_ms), beta)  # Cosine window
        window = np.zeros((len(IR[:])))  # Final window

        ms = 64
        # Sample of the initial peak
        peak = detect_peaks(IR, mph=(max(IR) * 0.9), threshold=0, edge='rising', show=False)
        if len(peak) > 1:
            peak = peak[0]
            # print('More than one peak at the IR')
        #        ind[x] = 0; # Window max from the beginning
        dt = (max(t_IR) / len(t_IR))  # Time axis resolution
        tt_ms = round((ms / 1000) / dt)  # Number of samples equivalent to 64 ms

        # Windows
        post_peak = np.zeros((len(IR[:])))
        pre_peak = np.zeros((len(IR[:])))
        win_cos = win.cosine(int(2 * tt_ms))  # Cosine window
        window = np.zeros((len(IR[:])))  # Final window
        ##
        # Cosine window pre peak
        win_cos_b = win.cosine(2 * peak + 1)
        pre_peak[0:int(peak)] = win_cos_b[0:int(peak)]
        pre_peak[int(peak)::] = 1

        # Cosine window post peak
        post_peak[int(peak):int(peak + tt_ms)] = win_cos[int(tt_ms):int(2 * tt_ms)] / max(
            win_cos[int(1 * tt_ms):int(2 * tt_ms)])  # Creating Hanning window array

        # Creating final window
        window[0:int(peak)] = pre_peak[0:int(peak)]
        #         window[0:int(peak)] = 1  # 1 from the beggining
        window[int(peak)::] = post_peak[int(peak)::]

    # Applying window
    IR_array = np.zeros((len(IR), 2))  # Creating matrix
    IR_array[:, 0] = IR[:]
    IR_array[:, 1] = IR[:] * window[:]  # FR and SBIR

    # Calculating FFT
    FFT_array = np.zeros((len(IR_array[:, 0]), len(IR_array[0, :])), dtype='complex')  # Creating matrix

    if ABEC is True:
        FFT_array_dB = np.zeros((round(len(IR_array[:, 0]) / 2) - delta_ABEC, len(IR_array[0, :])), dtype='complex')
        FFT_array_Pa = np.zeros((round(len(IR_array[:, 0]) / 2) - delta_ABEC, len(IR_array[0, :])), dtype='complex')
    else:
        FFT_array_dB = np.zeros((round(len(IR_array[:, 0]) / 2), len(IR_array[0, :])), dtype='complex')
        FFT_array_Pa = np.zeros((round(len(IR_array[:, 0]) / 2), len(IR_array[0, :])), dtype='complex')

    for i in range(0, len(IR_array[0, :])):
        iIR = IR_array[:, i]
        FFT_array[:, i] = 2 / len(iIR) * np.fft.fft(iIR)
        if ABEC is True:
            FFT_array_Pa[:, i] = FFT_array[delta_ABEC:round(len(iIR) / 2), i]
        else:
            FFT_array_Pa[:, i] = FFT_array[0:round(len(iIR) / 2), i]
    for i in range(0, len(IR_array[0, :])):
        if ABEC is True:
            FFT_array_dB[:, i] = 20 * np.log10(np.abs(FFT_array[delta_ABEC:int(len(FFT_array[:, i]) / 2),
                                                      i]) / 2e-5)  # applying log and removing aliasing and first 20 Hz
        else:
            FFT_array_dB[:, i] = 20 * np.log10(
                np.abs(FFT_array_Pa[:, i]) / 2e-5)  # applying log and removing aliasing and first 20 H

    if ABEC is False:
        freq_FFT = np.linspace(0, len(IR) / 2, num=int(len(IR) / 2))  # Frequency vector for the FFT
    else:
        freq_FFT = np.linspace(fmin, fmax, num=len(FFT_array_dB[:, 0]))  # Frequency vector for the FFT

    # View windowed Impulse Response in time domain:
    if winCheck is True:
        figWin = plt.figure(figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')
        win_index = 0
        plt.plot(t_IR, IR_array[:, win_index], linewidth=3)
        plt.plot(t_IR, IR_array[:, win_index + 1], '--', linewidth=5)
        plt.plot(t_IR, window[:] * (max(IR_array[:, 0])), '-.', linewidth=5)
        plt.title('Impulse Response Windowing', fontsize=20)
        plt.xlabel('Time [s]', fontsize=20)
        plt.xlim([t_IR[0], 0.12])#t_IR[int(len(t_IR) / 8)]])
        # plt.xticks(np.arange(t_IR[int(peak[0])], t_IR[int(len(t_IR))], 0.032), fontsize=15)
        plt.ylabel('Amplitude [-]', fontsize=20)
        plt.legend(['Modal IR', 'SBIR IR', 'Window'], loc='best', fontsize=20)
        plt.grid(True, 'both')
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()

    # Frequency Response and SBIR
    if spectraCheck is True:
        figSpectra = plt.figure(figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')

        if ABEC is False:
            plt.semilogx(freq_FFT[fmin:fmax + 1], FFT_array_dB[fmin:fmax + 1, 0], linewidth=3, label='Full spectrum')
            plt.semilogx(freq_FFT[fmin:fmax + 1], FFT_array_dB[fmin:fmax + 1, 1], '-.', linewidth=3, label='SBIR')
        elif ABEC is True:
            plt.semilogx(freq_FFT, FFT_array_dB[:, 0], linewidth=3, label='Full spectrum')
            plt.semilogx(freq_FFT, FFT_array_dB[:, 1], '-.', linewidth=3, label='SBIR')

        # plt.semilogx(freq_FFT[fmin:fmax + 1], 20 * np.log10(np.abs(FFT_array_Pa[fmin:fmax + 1, 1])/2e-5), ':',
        #              linewidth=3, label='SBIR 2')
        plt.legend(fontsize=15, loc='best')  # , bbox_to_anchor=(0.003, 0.13))
        # plt.title('Processed IR vs ABEC', fontsize=20)
        plt.xlabel('Frequency [Hz]', fontsize=20)
        plt.ylabel('SPL [dB ref. 20 $\mu$Pa]', fontsize=20)
        plt.gca().get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
        plt.gca().get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
        plt.gca().tick_params(which='minor', length=5)  # Set major and minor ticks to same length
        plt.xticks(fontsize=15)
        plt.grid(True, 'both')
        plt.xlim([fmin, fmax])
        plt.ylim([55, 105])
        plt.yticks(fontsize=15)
        plt.tight_layout(pad=3)
        plt.show()
    return freq_FFT, FFT_array_Pa, window
    # return freq_FFT[closest_node(freq_FFT,fmin):closest_node(freq_FFT,fmax)+1], FFT_array_Pa[closest_node(freq_FFT,fmin):closest_node(freq_FFT,fmax)+1,1], window



class IR(object):
    """Perform a room impulse response computation."""

    def __init__(self, sampling_rate, duration,
            minimum_frequency, maximum_frequency):
        """
        Setup the room impulse computation.

        Parameters
        ----------
        sampling_rate : integer
            Sampling rate in Hz for the time signal.
        duration: float
            Time in seconds until which to sample the room impulse response.
        minimum_frequency: float
            Minimum sampling frequency
        maximum_frequency: float
            Maximum sampling frequency

        """
        self._number_of_frequencies = int(round(sampling_rate * duration))
        self._sampling_rate = sampling_rate
        self._duration = duration
        self._frequencies = (sampling_rate * np.arange(self._number_of_frequencies) 
                / self._number_of_frequencies)
        self._timesteps = np.arange(self._number_of_frequencies) / sampling_rate

        self._maximum_frequency = maximum_frequency
        self._minimum_frequency = minimum_frequency

        self._frequency_filter_indices = np.flatnonzero(
                (self._frequencies <= self._maximum_frequency) & 
                (self._frequencies >= self._minimum_frequency))

        self._high_pass_frequency = 2 * minimum_frequency
        self._low_pass_frequency = 2 * maximum_frequency

        self._high_pass_order = 4
        self._low_pass_order = 4

        self._alpha = 0.18  # Tukey window alpha

        
    @property
    def number_of_frequencies(self):
        """Return number of frequencies."""
        return self._number_of_frequencies

    @property
    def sampling_rate(self):
        """Return sampling rate."""
        return self._sampling_rate

    @property
    def duration(self):
        """Return duration."""
        return self._duration

    @property
    def timesteps(self):
        """Return time steps."""
        return self._timesteps

    @property
    def frequencies(self):
        """Return frequencies."""
        return self._frequencies

    @property
    def filtered_frequencies(self):
        """Return the filtered frequencies."""
        return self.frequencies[
                self._frequency_filter_indices
                ]

    @property
    def maximum_frequency(self):
        """Return maximum frequency."""
        return self._maximum_frequency

    @property
    def minimum_frequency(self):
        """Return minimum frequency."""
        return self._minimum_frequency

    @property
    def high_pass_frequency(self):
        """Return high pass frequency."""
        return self._high_pass_frequency

    @high_pass_frequency.setter
    def high_pass_frequency(self, freq):
        """Set high pass frequency."""
        self._high_pass_frequency = freq

    @property
    def low_pass_frequency(self):
        """Return low pass frequency."""
        return self._low_pass_frequency

    @low_pass_frequency.setter
    def low_pass_frequency(self, freq):
        """Set low pass frequency."""
        self._low_pass_frequency = freq

    @property
    def high_pass_filter_order(self):
        """Return high pass filter order."""
        return self._high_pass_order

    @high_pass_filter_order.setter
    def high_pass_filter_order(self, order):
        """Set high pass filter order."""
        self._high_pass_order = order

    @property
    def low_pass_filter_order(self):
        """Return low pass filter order."""
        return self._low_pass_order

    @low_pass_filter_order.setter
    def low_pass_filter_order(self, order):
        """Set low pass filter order."""
        self._low_pass_order = order


    # def compute_room_impulse_response(
    #         self, values_at_filtered_frequencies):
    #     """
    #     Compute the room impulse response.

    #     Parameters
    #     ----------
    #     values_at_filtered_frequencies : array
    #         The frequency domain values to be transformed taken
    #         at the filtered frequencies.

    #     Output
    #     ------
    #     An array of approximate time values at the given time steps.
        
    #     """
    #     from scipy.signal import butter, freqz, tukey
    #     from scipy.fftpack import ifft
        
    #     b_high, a_high = butter(
    #             self.high_pass_filter_order,
    #             self.high_pass_frequency * 2 / self.sampling_rate, 
    #             'high')

    #     b_low, a_low = butter(
    #             self.low_pass_filter_order,
    #             self.low_pass_frequency * 2 / self.sampling_rate, 
    #             'low')

    #     high_pass_values = freqz(
    #             b_high, a_high, self.filtered_frequencies,
    #             fs=self.sampling_rate)[1]

    #     low_pass_values = freqz(
    #             b_low, a_low, self.filtered_frequencies,
    #             fs=self.sampling_rate)[1]

    #     butter_filtered_values = (values_at_filtered_frequencies * 
    #             np.conj(low_pass_values) * np.conj(high_pass_values))

    #     # windowed_values = butter_filtered_values * tukey(len(self.filtered_frequencies),
    #     #         min([self.maximum_frequency - self.low_pass_frequency,
    #     #              self.high_pass_frequency - self.minimum_frequency]) /
    #     #         (self.maximum_frequency - self.minimum_frequency))

    #     windowed_values = butter_filtered_values * tukey(len(self.filtered_frequencies), alpha=self._alpha)

    #     full_frequency_values = np.zeros(self.number_of_frequencies, dtype='complex128')
    #     full_frequency_values[self._frequency_filter_indices] = windowed_values
    #     full_frequency_values[-self._frequency_filter_indices] = np.conj(windowed_values)

    #     return ifft((full_frequency_values)) * self.number_of_frequencies
    
def fft(time_data, axis=0, crop=True):
    """
    Compute the FFT of a time signal.

    Parameters
    ----------
    time_data : array
        Time data array.
    axis : int, optional
        NumPy's axis option.
    crop : bool, optional
        Option the remove the second half of the FFT.

    Returns
    -------
    Frequency response array.
    """

    freq_data = 1 / len(time_data) * np.fft.fft(time_data, axis=axis)

    if crop:
        freq_data = freq_data[0:int(len(freq_data) / 2) + 1]

    return freq_data

def pressure2spl(pressure, ref=2e-5):
    """
    Computes Sound Pressure Level [dB] from complex pressure values [Pa].

    Parameters
    ----------
    pressure: array
        Complex pressure array.

    Returns
    -------
    SPL array.
    """
    spl = 10 * np.log10(0.5 * pressure * np.conj(pressure) / ref ** 2)

    return np.real(spl)

def impulse_response_calculation(freq, ir, values_at_freq, freq_filter_indices, tukey, alpha, rolling_window,
                                 cut_sample, filter_sample, n, return_id, high_pass_values, low_pass_values,
                                 base_fontsize=15, linewidth=2, figsize=(16, 20)):
    """
    Auxiliary plot to view impulse response calculation results, windows and filters.

    Parameters
    ----------
    freq : array
        Frequency vector.
    ir : list
        List containing calculated IRs.
    values_at_freq : array
        Complex pressure values at the receiver.
    freq_filter_indices : array
        Indices of the frequency range of analysis.
    tukey : array
        Tukey window used to filter the noise at the end of the IR during post-processing.
    alpha : float
        Alpha value of the pre-processing Tukey window.
    rolling_window : iterable
        List of rolling windows.
    cut_sample : int
        Sample of the window of lowest energy.
    filter_sample : int
        Sample at which the Tukey window will start to remove the noise at the end.
    n : int
        Size of the rolling windows in samples.
    return_id : int
        Index of which IR from the IR list will be returned.
    high_pass_values : array
        High-pass filter frequency response.
    low_pass_values : array
        Low-pass filter frequency response.
    base_fontsize : int
        Base font size.
    linewidth : int
        Plot line width.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    Matplotlib figure, Gridspec object and list of Matplotlib axes.
    """

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 1)
    ax = [plt.subplot(gs[i, 0]) for i in range(4)]

    i = 0

    ax[i].set_title(f"Impulse Response", size=base_fontsize + 2, fontweight="bold", loc="left")
    ax[i].set_xlabel("Samples [n]", size=base_fontsize)
    ax[i].set_ylabel("Amplitude [-]", size=base_fontsize)
    ax[i].plot(ir[0], alpha=0.3, linewidth=linewidth, label="Filtered Input Data")
    ax[i].axvline(filter_sample * n, color="b", label="Filter start point")
    ax[i].plot(tukey * max(ir[0]), label="Filter window", linewidth=linewidth)
    for r in range(len(rolling_window) + 1):
        ax[i].axvline(r * n, color="k", zorder=0, alpha=0.3, linewidth=linewidth, linestyle=":",
                      label="Rolling windows" if r == 0 else None)
    ax[i].axvline(cut_sample * n, color="r", label="Lowest energy", linewidth=linewidth, linestyle="--")
    ax[i].plot(ir[return_id], alpha=0.7, linewidth=linewidth, label="Processed IR")
    i += 1

    ax[i].set_title(f"Frequency Response", size=base_fontsize + 2, fontweight="bold", loc="left")
    ax[i].set_xlabel("Frequency [Hz]", size=base_fontsize)
    ax[i].set_ylabel("Amplitude [dB]", size=base_fontsize)
    ax[i].plot(freq, fd.p2SPL(np.abs(values_at_freq)), label="Input Data", linewidth=linewidth)
    ax[i].plot(freq, fd.p2SPL(np.abs(fft(ir[return_id])[freq_filter_indices])),
               label="Processed IR", linewidth=linewidth, linestyle=":")
    i += 1

    ax[i].set_title(f"Phase Response", size=base_fontsize + 2, fontweight="bold", loc="left")
    ax[i].set_xlabel("Frequency [Hz]", size=base_fontsize)
    ax[i].set_ylabel("Angle [deg]", size=base_fontsize)
    ax[i].plot(freq, np.rad2deg(np.angle(values_at_freq)), label="Input Data", linewidth=linewidth)
    ax[i].plot(freq, np.rad2deg(np.angle(fft(ir[return_id])[freq_filter_indices])),
               label="Processed IR",
               linewidth=linewidth)
    i += 1


    ax[i].set_title(f"Filters and windows response", size=base_fontsize + 2, fontweight="bold", loc="left")
    ax[i].set_xlabel("Frequency [Hz]", size=base_fontsize)
    ax[i].set_ylabel("Amplitude [dBFS]", size=base_fontsize)
    ax[i].plot(freq, pressure2spl(abs(high_pass_values), ref=1), label="High-pass",
               linewidth=linewidth)
    ax[i].plot(freq, pressure2spl(abs(low_pass_values), ref=1), label="Low-Pass",
               linewidth=linewidth)
    ax[i].plot(freq, pressure2spl(abs(signal.tukey(len(freq), alpha)), ref=1),
               label="Tukey", linewidth=linewidth)
    i += 1

    for j in range(i):
        ax[j].legend(fontsize=base_fontsize - 2, loc="best", ncol=6)
        ax[j].grid("minor")
        ax[j].tick_params(axis='both', which='both', labelsize=base_fontsize - 3)

    gs.tight_layout(fig, pad=1)

    return fig, gs, ax

class Domain:
    """Acoustical domain properties and methods."""
    def __init__(self, fmin, fmax, tmax, fs=44100):
        """
        Parameters
        ----------
        fmin : float
            Minimum sampling frequency.
        fmax : float
            Maximum sampling frequency.
        tmax : float
            Time in seconds until which to sample the room impulse response.
        fs : int, optional
            Sampling rate in Hz for the time signal.
        """
        self._fmin = fmin
        self._fmax = fmax
        self._tmax = tmax
        self._fs = fs
        self._high_pass_freq = 2 * self.fmin
        self._low_pass_freq = 2 * self.fmax
        self._high_pass_order = 4
        self._low_pass_order = 4
        self._freq_overhead = [2, 1.2]
        self._alpha = None
        self._air_prop = fd.AirProperties()

    @property
    def air_prop(self):
        """Return air properties dictionary."""
        return self._air_prop.standardized_c0_rho0()

    @property
    def num_freq(self):
        """Return number of frequencies."""
        return int(round(self.fs * self.tmax))

    @property
    def fs(self):
        """Return sampling rate."""
        return self._fs

    @property
    def tmax(self):
        """Return time duration."""
        return self._tmax

    @property
    def time(self):
        """Return time steps."""
        return np.arange(self.num_freq, dtype="float64") / self.fs

    @property
    def all_freqs(self):
        """Return frequencies."""
        return self.fs * np.arange(self.num_freq, dtype="float64") / self.num_freq

    @property
    def df(self):
        """Return frequency resolution."""
        return self.fs / self.num_freq

    # @property
    # def freq_overhead(self):
    #     """Return the lower and upper overhead range factors."""
    #     return self._freq_overhead

    @property
    def freq_filter_indices(self):
        """Return the indices of the filtered frequencies."""
        # return np.flatnonzero((self.all_freqs >= self.fmin / self.freq_overhead[0])
                               # & (self.all_freqs <= self.fmax * self.freq_overhead[1]))

        # return np.flatnonzero(((self.all_freqs >= self.fmin)
        #                        & (self.all_freqs <= self.fmax)))

        return np.flatnonzero(((self.all_freqs >= self.fmin) & (self.all_freqs <= self.fmax)))

    @property
    def freq(self):
        """Return the filtered frequencies."""
        return self.all_freqs[self.freq_filter_indices]

    @property
    def w0(self):
        """Return the filtered frequencies."""
        return 2 * np.pi * self.freq

    @property
    def fmin(self):
        """Return minimum frequency."""
        return self._fmin

    @property
    def fmax(self):
        """Return maximum frequency."""
        return self._fmax

    @property
    def df(self):
        """Return frequency resolution."""
        return 1 / self.tmax

    @property
    def high_pass_freq(self):
        """Return high pass frequency."""
        return self._high_pass_freq

    @high_pass_freq.setter
    def high_pass_freq(self, freq):
        """Set high pass frequency."""
        self._high_pass_freq = freq

    @property
    def low_pass_freq(self):
        """Return low pass frequency."""
        return self._low_pass_freq

    @low_pass_freq.setter
    def low_pass_freq(self, freq):
        """Set low pass frequency."""
        self._low_pass_freq = freq

    @property
    def high_pass_filter_order(self):
        """Return high pass filter order."""
        return self._high_pass_order

    @high_pass_filter_order.setter
    def high_pass_filter_order(self, order):
        """Set high pass filter order."""
        self._high_pass_order = order

    @property
    def low_pass_filter_order(self):
        """Return low pass filter order."""
        return self._low_pass_order

    @low_pass_filter_order.setter
    def low_pass_filter_order(self, order):
        """Set low pass filter order."""
        self._low_pass_order = order

    @property
    def alpha(self):
        """Return Tukey window alpha value."""
        return self._alpha

    @alpha.setter
    def alpha(self, alphaValue):
        """Set Tukey window alpha value."""
        self._alpha = alphaValue

    def bands(self, n_oct=3):
        """Return octave frequency bands."""
        # bands = pytta.utils.freq.fractional_octave_frequencies(nthOct=n_oct)  # [band_min, band_center, band_max]
        return pytta.utils.freq.fractional_octave_frequencies(freqRange=(self.fmin, self.fmax), nthOct=n_oct)[:, 1]
        # return bands[np.argwhere((bands[:, 1] >= min(self.freq)) & (bands[:, 1] <= max(self.freq)))[:, 0]]

    def compute_impulse_response(self, values_at_freq, alpha=None, auto_roll=False, auto_window=True, irr_filters=False,
                                 view=False):
        """
        Compute the room impulse response.

        Parameters
        ----------
        values_at_freq : array
            The frequency domain values to be transformed taken at the filtered frequencies.
        alpha : float or None
            Tukey window alpha value. If 'None' is passed it will be automatically calculated.
        auto_rool : bool, optional
            Option to automatically roll the impulse response to ensure no noise at the end.
        auto_window : bool, optional
            Option to automatically filter the impulse response to ensure no noise at the end.
        irr_filters : bool, optionak
            Option to use additional low and high-pass filters during the preconditioning of the input data.
        view : bool, optional
            Option to show plot with different parameters for analysis.

        Returns
        ------
        An array of approximate time values at the given time steps.
        """

        # Applying low and high-pass filters
        """
        This minimizes the phase differences from the input pressure data to the FFT of the  computed impulse response.
        Filters of order higher than 4 are unstable in Scipy, consider replacing this by FIR filters in the future.
        """
        b_high, a_high = signal.butter(self.high_pass_filter_order, self.high_pass_freq * 2 / self.fs, "high")
        b_low, a_low = signal.butter(self.low_pass_filter_order, self.low_pass_freq * 2 / self.fs, "low")
        _, high_pass_values = signal.freqz(b_high, a_high, self.freq, fs=self.fs)
        _, low_pass_values = signal.freqz(b_low, a_low, self.freq, fs=self.fs)
        butter_filtered_values = (values_at_freq * np.conj(low_pass_values) * np.conj(high_pass_values))

        # Applying Tukey window
        self.alpha = max([self.fmax - self.low_pass_freq, self.high_pass_freq - self.fmin]) / (
                    self.fmax - self.fmin) if alpha is None else alpha
        windowed_values = butter_filtered_values * signal.tukey(len(self.freq),
                                                                self.alpha) if irr_filters \
            else values_at_freq * signal.tukey(len(self.freq), self.alpha)
        full_freq_values = np.zeros(self.num_freq, dtype="complex64")
        full_freq_values[self.freq_filter_indices] = windowed_values
        full_freq_values[-self.freq_filter_indices] = np.conj(windowed_values)

        # Array containing the IR at different calculation points - [raw, rolled, filtered]
        ir = [np.real(np.fft.ifft(full_freq_values) * self.num_freq) for _ in range(3)]
        return_id = 0

        # Identifying noise at the end
        parts = 100
        n = int(self.fs / parts)  # Dividing the impulse response into equal parts
        rolling_window = list(more_itertools.windowed(ir[0], step=n, n=n, fillvalue=0))  # Creating the rolling window
        delta_amp = [np.sum(np.abs(win) ** 2) for win in rolling_window]  # Getting the total energy of each slice

        # Rolling the ir to remove move noise at the end
        cut_sample = delta_amp.index(np.min(delta_amp)) + 1  # Finding the slice with the smallest amplitude variation
        cut_size = (len(rolling_window) - cut_sample) * n
        ir[1] = np.roll(ir[1], cut_size)  # Rolling the impulse response

        # Filter the IR to remove noise at the end
        filter_sample = delta_amp.index(np.min(delta_amp)) - int(
            parts / (parts * 0.75))  # Finding the slice with the smallest amplitude variation
        filter_size = (len(rolling_window) - filter_sample) * n
        tukey = np.concatenate((np.ones(filter_sample * n), signal.tukey(filter_size * 2, 1)[filter_size::]))
        ir[2] = ir[2] * tukey

        if auto_roll:
            return_id = 1
        if auto_window:
            return_id = 2

        # Plotting
        if view:
            _, _, _ = impulse_response_calculation(self.freq, ir, values_at_freq, self.freq_filter_indices, tukey,
                                                        self.alpha, rolling_window, cut_sample, filter_sample, n,
                                                        return_id, high_pass_values, low_pass_values)

        return ir[return_id]