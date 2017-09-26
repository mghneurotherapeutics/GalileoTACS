import h5py
import numpy as np


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
