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


def compute_toi_power(power, times, freqs, band, toi):

    # extract band power
    band_mask = np.where(np.logical_and(freqs >= band[0], freqs <= band[1]))
    power = np.mean(power[:, band_mask, :].squeeze(), axis=1)

    # average over array
    power = power.mean(axis=0)

    # average over time of interest
    toi_mask = np.where(np.logical_and(times >= toi[0], times <= toi[1]))
    power = power[toi_mask].mean()

    return power


def compute_bootstrap_p_value(power, bootstrap_dist, times, toi):

    # reduce to toi
    toi_mask = np.where(np.logical_and(times >= toi[0], times <= toi[1]))
    power_toi = power[toi_mask].mean()
    bootstrap_toi_dist = np.sort(bootstrap_dist[:, toi_mask].mean(axis=-1))

    center_dist = bootstrap_toi_dist - power_toi

    p_num = np.sum(np.abs(center_dist) >= np.abs(power_toi)) + 1.
    p_denom = len(bootstrap_toi_dist) + 1.

    return p_num / p_denom


def compute_permutation_toi(perm_num, all_conditions_power, trial_indices,
                            permutation_indices, times, freqs, alpha, beta,
                            toi, baseline, conditions):

    print(perm_num)

    # collect power across conditions into single array
    # we downsample to match trial sizes
    power = []
    for c in conditions:
        trial_ix = trial_indices[c][perm_num, :]
        power.append(all_conditions_power[c][trial_ix, :, :, :].squeeze())
    power = np.vstack(power)

    # permute the data
    perm_ix = permutation_indices[perm_num, :]
    power = power[perm_ix, :, :, :].squeeze()

    # baseline normalize
    cond_len = power.shape[0] / 2
    tmp = []
    tmp.append(baseline_normalize(power[:cond_len, :], baseline, times))
    tmp.append(baseline_normalize(power[cond_len:, :], baseline, times))
    power = tmp

    # compute toi power difference
    diffs = []
    for band in [alpha, beta]:
        c1_toi = compute_toi_power(power[0], times, freqs,
                                   band, toi)
        c2_toi = compute_toi_power(power[1], times, freqs,
                                   band, toi)
        diffs.append(c1_toi - c2_toi)

    return diffs

def compute_bootstrap_sample(bootstrap_ix, power, times, freqs, alpha, beta,
                             baseline):

    # permute the data
    power = power[bootstrap_ix, :, :, :].squeeze()

    # baseline normalize
    power = baseline_normalize(power, baseline, times)

    # average over array
    power = power.mean(axis=0)

    # compute band power
    output = []
    for band in [alpha, beta]:
        band_mask = np.where(np.logical_and(freqs >= band[0], freqs <= band[1]))
        output.append(np.mean(power[band_mask, :].squeeze(), axis=0))

    return output

def bootstrap_standard_error(data, num_bootstraps, bootstrap_ix):
    """
    """

    bootstrap_indices = np.random.choice(data.shape[bootstrap_ix],
                                         size=(num_bootstraps,
                                               data.shape[bootstrap_ix]),
                                         replace=True)

    bootstrap_samples = np.zeros([num_bootstraps] + [dim for dim in data.shape])
    for i in range(num_bootstraps):
        bootstrap_samples[i, :] = data[bootstrap_indices[i, :], :]

    bootstrap_power = bootstrap_samples.mean(axis=bootstrap_ix + 1)
    bootstrap_power = bootstrap_power / bootstrap_power.sum(axis=-1)[:, np.newaxis]
    bootstrap_std_err = bootstrap_power.std(axis=0)
    return bootstrap_std_err


def pre_compute_permutation_indices(tests, sample_sizes, num_permutations,
                                    seed=2129):

    np.random.seed(seed)

    permutation_indices = {}

    for ss, t in zip(sample_sizes, tests):
        permutations = np.zeros((num_permutations, ss * 2), dtype=np.int32)
        ix = np.arange(ss * 2)
        for i in range(num_permutations):
            np.random.shuffle(ix)
            permutations[i, :] = ix
        permutation_indices[t] = permutations

    return permutation_indices


def pre_compute_subsample_indices(conditions, original_sample_sizes,
                                  desired_sample_sizes, num_permutations,
                                  seed=2129):

    np.random.seed(seed)
    trial_indices = {}

    for oss, dss, c in zip(original_sample_sizes, desired_sample_sizes,
                           conditions):
        trial_indices[c] = np.zeros((num_permutations, dss), dtype=np.int32)
        for i in range(num_permutations):
            trial_indices[c][i, :] = np.random.choice(oss, size=dss,
                                                      replace=False)

    return trial_indices

