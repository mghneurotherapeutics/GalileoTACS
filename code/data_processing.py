import h5py
import numpy as np
import glob
from mne import create_info
from mne.io import RawArray


def extract_blackrock_info(mat_file, blackrock_type):
    """ Extracts basic recording info from a blacrock extracted mat file.

    Extracts the data, sampling rate, channel names, and digital to
    analog conversion factor from a blackrock extracted .mat file. hp5y was
    required instead of scipy.loadmat due to the large .mat file size.

    Args:
        mat_file: string of filename representing a .mat file extracted from a
            blackrock .ns2 or .ns5 file using OpenNSx
        blackrock_type: a string either 'ns2' or 'ns5' denoting which type
            of recording the .mat file contains

    Returns:
        a dictionary containing the data, sampling rate,
        channel names, and digital to analog conversion factor
    """

    info = {}
    file_obj = h5py.File(mat_file)
    struct = file_obj[blackrock_type.upper()]

    if 'saline1' in mat_file:
        data = [file_obj[struct['Data'][0, 0]], file_obj[struct['Data'][1, 0]]]
        info['data'] = np.concatenate(data, axis=0).T
    else:
        info['data'] = np.array(struct['Data']).T
    info['srate'] = struct['MetaTags']['SamplingFreq'][0][0]

    # extract the digital to analog conversion factor as the ratio
    # between the analog range and digital range
    max_av = file_obj[struct['ElectrodesInfo']
                            ['MaxAnalogValue'][0][0]][0][0]
    max_dv = file_obj[struct['ElectrodesInfo']
                            ['MaxDigiValue'][0][0]][0][0]
    info['dac_factor'] = float(max_av) / max_dv

    # extract the channel names
    ch_name_datasets = [file_obj[ref[0]] for ref in
                        np.array(struct['ElectrodesInfo']['Label'])]
    ch_names = [u''.join(unichr(c) for c in l if c)
                for l in ch_name_datasets]
    # replace 'elec1-84' with ref since this was only done in some files
    info['ch_names'] = [u'ref' if c == "elec1-84" else c for c in ch_names]

    return info


def create_mne_raw(blackrock_info):
    """ Creates an MNE-Python raw object given a dictionary containing
    recording information extracted from the blacrock mat file.

    Args:
        blackrock_info: dictionary containing the data, channel names,
            sampling rate, and dac factor

    Returns:
        an MNE-Python raw object for the data
    """

    # create the MNE info object
    ch_types = ['eeg'] * len(blackrock_info['ch_names']) + ['stim']
    blackrock_info['ch_names'].append("STIM")
    mne_info = create_info(blackrock_info['ch_names'],
                           blackrock_info['srate'], ch_types)

    # take the recorded data and add a row of 0's to represent
    # the stim channel without events yet
    num_samples = blackrock_info['data'].shape[-1]
    blackrock_info['data'] = np.vstack((blackrock_info['data'],
                                        np.zeros(num_samples)))

    # convert from digitized units to microvolts
    blackrock_info['data'] *= blackrock_info['dac_factor']

    # create MNE Raw object
    raw = RawArray(blackrock_info['data'], mne_info, verbose=False)

    return raw


def create_events_square_wave(events):
    """ Takes an MNE events array consisting of pairs of onset and offset
    events and interpolates new events between these onset and offset events to
    form a "square wave" for placement into an MNE stim channel.

    Args:
        events: An MNE events array consisting of onset and offset
            paired events.

    Returns:
        The new events array with samples between onset and offset
            events filled with events.
    """
    filled_events = []

    i = 0
    while i < events.shape[0]:
        onset, offset = events[i, 0], events[i + 1, 0]
        for j in range(onset, offset + 1):
            filled_events.append([j, 0, 1])
        i += 2

    return np.array(filled_events)


def load_power_data(exp, condition, typ='ns2'):
    """ Loads all tfr power for a given experiment and condition.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'
        condition: The condition to collect data for.
            'Open', 'Closed', or 'Brain'
        typ: The type of recording file to collect. 'ns2' or 'ns5'.

    Returns:
        A tuple containing the power data across all dates in a single array,
        a list of channel names, a list of time labels, and a list of
        frequency labels.
    """

    file = '../data/power/%s_%s_*_raw_power.npz' % (typ, condition)
    fnames = sorted(glob.glob(file))

    if exp == 'saline':
        fnames = [f for f in fnames if 'saline' in f]
    else:
        fnames = [f for f in fnames if 'saline' not in f]

    tmp = np.load(fnames[0])
    chs = tmp['chs']
    times = tmp['times']
    freqs = tmp['freqs']

    power = [np.load(f)['data'] for f in fnames]
    power = np.concatenate(power, axis=0)

    return power, chs, times, freqs


def baseline_normalize(power, baseline, times):
    """ Baseline normalizes raw tfr power data according to
    a slightly modified version of the normalization procedure suggested by
    Grandchamp and Delorme, 2011.

    First, we divide the power data in each trial by the median of the
    power across the entire trial (excluding the stimulation period and 0.5
    seconds of buffer around the stimulation period). Then, we take the median
    across all trials and divide the median power by the median of the
    pre-stimulation baseline period. Finally, we log transform and multipy
    by 10 to get a decibel representation.

    Args:
        power: # trials x # chs x # freqs x # time points array
            containing TFR power
        baseline: tuple delimiting the time boundaries of baseline period
        times: a list of time labels for each sample

    Returns:
        The modified tfr power array now baseline normalized (# chs x
        # freqs x # time points)
    """

    # first normalize by the median of the power across the entire trial
    # excluding the stimulation period and stimulation edge artifacts
    trial_mask = np.where(np.logical_or(times <= -.5, times >= 10.5))[0]
    trial_norm = np.median(power[:, :, :, trial_mask],
                           axis=-1)[:, :, :, np.newaxis]
    power /= trial_norm

    # median across trials
    power = np.median(power, axis=0)

    # normalize by median of pre-stimulation baseline period
    bl_mask = np.where(np.logical_and(times >= baseline[0],
                                      times <= baseline[1]))[0]
    bl_norm = np.median(power[:, :, bl_mask], axis=-1)[:, :, np.newaxis]
    power /= bl_norm

    # log transform and scale
    power = 10 * np.log10(power)

    return power


def reduce_band_power(power, freqs, band, axis):
    """ Averages frequency content within a given frequency band range.

    Args:
        power: array containing tfr power
        freqs: list of frequencies contained in the tfr power array
        band: tuple containing the frequency band limits
        axis: the axis containing the frequency data

    Returns:
        Returns a band power array where the frequency axis has been averaged
        within the range supplied by band.
    """
    band_mask = np.where(np.logical_and(freqs >= band[0], freqs <= band[1]))[0]
    power = np.take(power, band_mask, axis=axis).mean(axis=axis)
    return power


def reduce_toi_power(power, times, toi, axis):
    """ Averages across time withing a given period of interest.

    Args:
        power: array containing tfr power
        times: list of time labels for each sample
        toi: tuple containing the limits of the time period of interest
        axis: the axis containing the time data

    Returns:
        Returns a power array where the time axis has been averaged
        within the range supplied by toi.
    """
    toi_mask = np.where(np.logical_and(times >= toi[0], times <= toi[1]))[0]
    power = np.take(power, toi_mask, axis=axis).mean(axis=axis)
    return power


def reduce_array_power(power, chs, bad_chs, array, axis):
    """ Averages across channels withing a given array.

    Args:
        power: array containing tfr power
        chs: list of channel names
        bad_chs: bad channels not to be included in average
        array: which recording array to average over
        axis: the axis containing the ch info

    Returns:
        Returns a power array where the channel axis has been averaged
        within the selected chs supplied by array and bad_chs.
    """

    arr_base = 'elec%s' % array
    ch_mask = [ix for ix in np.arange(len(chs)) if arr_base in chs[ix] and
               chs[ix] not in bad_chs]
    power = np.take(power, ch_mask, axis=axis).mean(axis=axis)
    return power
