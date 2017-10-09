import h5py
import numpy as np
import glob
from mne import create_info
from mne.io import RawArray


def extract_blackrock_info(mat_file, blackrock_type):
    """ Extracts the data, sampling rate, channel names, and digital to
    analog conversion factor from a blackrock extracted .mat file. hp5y was
    required instead of scipy.loadmat due to the large .mat file size.

    Inputs:
    - mat_file: string of filename representing a .mat file extracted from a
    blackrock .ns2 or .ns4 file using OpenNSx

    Outputs:
    - info: dictionary containing the data, sampling rate,
    channel names, and digital to analog conversion factor
    """

    info = {}
    file_obj = h5py.File(mat_file)
    struct = file_obj[blackrock_type.upper()]

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
    recording information extracted from the blacrock datatype.

    Inputs:
    - blackrock_info: dictionary containing the data, channel names,
    sampling rate, and dac factor

    Outputs:
    - raw: MNE-Python raw object for the data
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


def load_all_data(exp, condition):

    file = './power/raw/ns2_%s_*_raw_power.npz' % condition
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


def create_events_square_wave(events):
    """ Takes an MNE events array consisting of pairs of onset and offset
    events and interpolates new events between these onset and offset events to
    form a "square wave" for placement into an MNE stim channel.

    Inputs:
    - events: An MNE events array consisting of onset and offset paired events.
    Must be even length.

    Outputs:
    - filled_events: The new events array with samples between onset and offset
    events filled with events.
    """
    filled_events = []

    i = 0
    while i < events.shape[0]:
        onset, offset = events[i, 0], events[i + 1, 0]
        for j in range(onset, offset + 1):
            filled_events.append([j, 0, 1])
        i += 2
    filled_events = np.array(filled_events)
    return filled_events


# Data Processing Utilities


def baseline_normalize(power, baseline, times):
    """ Baseline normalizes according to the methodology described in
    Grandchamp and Delorme, 2011 with the exception that we exclude the
    stimulation from the initial full trial normalization step.

    Inputs:
    - power: # trials x # freqs x # time points array containing TFR power
    - baseline: tuple delimiting the time boundaries of baseline period
    - times: a list of time labels for each sample

    Outpus:
    - power: modified tfr power now baseline normalized
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
    band_mask = np.where(np.logical_and(freqs >= band[0], freqs <= band[1]))[0]
    power = np.take(power, band_mask, axis=axis).mean(axis=axis)
    return power


def reduce_toi_power(power, times, toi, axis):
    toi_mask = np.where(np.logical_and(times >= toi[0], times <= toi[1]))[0]
    power = np.take(power, toi_mask, axis=axis).mean(axis=axis)
    return power


def reduce_array_power(power, chs, bad_chs, axis):

    ch_mask = [ix for ix in np.arange(len(chs)) if 'elec1' in chs[ix] and
                                                   chs[ix] not in bad_chs]
    power = np.take(power, ch_mask, axis=axis).mean(axis=axis)
    return power