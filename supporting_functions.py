import h5py
import numpy as np
import glob
import json
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

## Data Cleaning and Extraction Utilities

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


def load_all_data(exp, condition):

    fnames = sorted(glob.glob('./power/raw/ns2_%s_*_raw_power.npz' % condition))
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


# Bootstrap Functions

def pre_compute_bootstrap_indices(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    np.random.seed(config['random_seed'])

    bootstrap_indices = {}
    for ss, c in zip(config['%s_sample_sizes' % exp], config['conditions']):
        bootstrap_indices[c] = np.random.choice(ss,
                                                size=(config['num_bootstraps'],
                                                      ss),
                                                replace=True)
    f = './stats/%s_experiment/condition_bootstrap_indices.npz' % exp
    np.savez_compressed(f, Open=bootstrap_indices['Open'],
                        Closed=bootstrap_indices['Closed'],
                        Brain=bootstrap_indices['Brain'],
                        num_samples=config['num_bootstraps'])


def compute_bootstrap_p_value(power, bootstrap_dist, times, toi):

    # reduce to toi
    bootstrap_dist = reduce_toi_power(bootstrap_dist, times, toi, axis=-1)
    power = reduce_toi_power(power, times, toi, axis=-1)

    # sort by toi value
    bootstrap_dist = np.sort(bootstrap_dist)

    # center the distribution to make it a "null hypothesis"
    bootstrap_dist = bootstrap_dist - power

    # compute the p-value as the percentage of boostrap values larger
    # in absolute value than the
    p_num = np.sum(np.abs(bootstrap_dist) >= np.abs(power)) + 1.
    p_denom = len(bootstrap_dist) + 1.

    return p_num / p_denom


def compute_bootstrap_sample(bootstrap_ix, power, times, freqs, chs, config):

    # resample the data
    power = power[bootstrap_ix, :, :, :].squeeze()

    # baseline normalize
    power = baseline_normalize(power, config['baseline'], times)

    # reduce over array
    power = reduce_array_power(power, chs, config['bad_chs'], axis=0)

    # reduce over band
    output = []
    for band in ['alpha', 'beta']:
        output.append(reduce_band_power(power, freqs, config[band], axis=0))

    return output


def compute_bootstrap_distribution(exp):

    global power, times, freqs, chs, bootstrap_cond_ix, config

    # load in configurations
    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    # load in pre-computes bootstrap re-sample indices
    f = './stats/%s_experiment/condition_bootstrap_indices.npz' % exp
    bootstrap_indices = np.load(f)
    num_bootstrap_samples = bootstrap_indices['num_samples']

    for condition in config['conditions']:

        print('Computing Bootstrap Distribution for Condition: %s' % condition)

        power, chs, times, freqs = load_all_data(exp, condition)

        # compute the base band power
        base_ix = np.arange(power.shape[0])
        alpha_power, beta_power = compute_bootstrap_sample(base_ix, power,
                                                           times, freqs, chs,
                                                           config)

        # loop through all bootstrap samples in parallel
        bootstrap_cond_ix = bootstrap_indices[condition]
        bootstrap_samples = Parallel(n_jobs=config['n_jobs'])(delayed(compute_bootstrap_wrapper)(ix) for ix in range(num_bootstrap_samples))

        # collect all the bootstrap samples into single matrix
        alpha_bootstrap_samples = np.vstack([s[0] for s in bootstrap_samples])
        beta_bootstrap_samples = np.vstack([s[1] for s in bootstrap_samples])

        # compute p-values
        alpha_p = compute_bootstrap_p_value(alpha_power,
                                            alpha_bootstrap_samples,
                                            times, config['toi'])
        beta_p = compute_bootstrap_p_value(beta_power,
                                           beta_bootstrap_samples,
                                           times, config['toi'])

        # save
        f = './stats/%s_experiment/%s_bootstrap_info.npz' % (exp, condition)
        np.savez_compressed(f, alpha=alpha_power, beta=beta_power,
                            alpha_dist=alpha_bootstrap_samples,
                            beta_dist=beta_bootstrap_samples,
                            alpha_p=alpha_p, beta_p=beta_p, times=times)


# simple wrapper function to allow parallel computation
def compute_bootstrap_wrapper(ix):
    return compute_bootstrap_sample(bootstrap_cond_ix[ix], power,
                                    times, freqs, chs, config)


def simple_bootstrap(data, num_bootstraps, axis):
    """
    """

    bootstrap_indices = np.random.choice(data.shape[axis],
                                         size=(num_bootstraps,
                                               data.shape[axis]),
                                         replace=True)

    bootstrap_samples = np.zeros([num_bootstraps] +
                                 [dim for dim in data.shape])
    for i in range(num_bootstraps):
        bootstrap_samples[i, :] = data[bootstrap_indices[i, :], :]

    return bootstrap_samples


# Permutation Testing Functions


def compute_permutation_sample(perm_num, all_conditions_power, trial_indices,
                               permutation_indices, times, freqs, chs, config,
                               comp):

    # collect power across conditions into single array
    # we downsample to match trial sizes
    power = []
    for c in comp:
        if perm_num != -1:
            trial_ix = trial_indices[c][perm_num, :]
            power.append(all_conditions_power[c][trial_ix, :, :, :].squeeze())
        else:
            power.append(all_conditions_power[c])
    power = np.vstack(power)

    # permute the data
    if perm_num != -1:
        perm_ix = permutation_indices[perm_num, :]
        power = power[perm_ix, :, :, :].squeeze()

    # baseline normalize each condition separately
    if perm_num != -1:
        cond_len = power.shape[0] / 2
    else:
        cond_len = all_conditions_power[comp[0]].shape[0]
    tmp = []
    tmp.append(baseline_normalize(power[:cond_len, :], config['baseline'],
                                  times))
    tmp.append(baseline_normalize(power[cond_len:, :], config['baseline'],
                                  times))
    power = tmp

    # reduce over array
    power[0] = reduce_array_power(power[0], chs, config['bad_chs'], axis=0)
    power[1] = reduce_array_power(power[1], chs, config['bad_chs'], axis=0)

    # compute toi band power difference
    diffs = []
    for band in [config['alpha'], config['beta']]:

        # reduce to band
        c1_power = reduce_band_power(power[0], freqs, band, axis=0)
        c2_power = reduce_band_power(power[1], freqs, band, axis=0)

        # reduce over time
        c1_power = reduce_toi_power(c1_power, times, config['toi'], axis=0)
        c2_power = reduce_toi_power(c2_power, times, config['toi'], axis=0)

        diffs.append(c1_power - c2_power)

    return diffs


def compute_permutation_p_value(base_power, permutation_dist):
    num = np.sum(np.abs(np.array(permutation_dist)) >= np.abs(base_power)) + 1.
    denom = len(permutation_dist) + 1.
    return num / denom


def pre_compute_permutation_indices(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    tests = ["Open-Closed", "Open-Brain", "Brain-Closed"]

    np.random.seed(config['random_seed'])

    permutation_indices = {}

    for ss, t in zip(config['%s_sample_sizes' % exp], tests):
        permutations = np.zeros((config['num_permutations'], ss * 2),
                                dtype=np.int32)
        ix = np.arange(ss * 2)
        for i in range(config['num_permutations']):
            np.random.shuffle(ix)
            permutations[i, :] = ix
        permutation_indices[t] = permutations

    np.savez_compressed("./stats/%s_experiment/condition_permutation_indices.npz" % exp,
                        Open_Closed=permutation_indices["Open-Closed"],
                        Open_Brain=permutation_indices["Open-Brain"],
                        Brain_Closed=permutation_indices["Brain-Closed"],
                        num_permutations=config['num_permutations'])


def pre_compute_subsample_indices(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    np.random.seed(config['random_seed'])

    trial_indices = {}

    sample_sizes = config['%s_sample_sizes' % exp]
    for oss, dss, c in zip(sample_sizes, [min(sample_sizes)] * 3,
                           config['conditions']):
        trial_indices[c] = np.zeros((config['num_permutations'], dss),
                                    dtype=np.int32)
        for i in range(config['num_permutations']):
            trial_indices[c][i, :] = np.random.choice(oss, size=dss,
                                                      replace=False)

    np.savez_compressed('./stats/%s_experiment/condition_subsample_indices.npz' % exp,
                        Closed=trial_indices['Closed'],
                        Brain=trial_indices['Brain'],
                        Open=trial_indices['Open'])
    return trial_indices


def compute_permutation_distributions(exp):

    global all_conditions_power, times, freqs, chs, trial_indices, permutation_ix, comp, config

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    # load pre-sampled indices
    permutation_indices = np.load('./stats/%s_experiment/condition_permutation_indices.npz' % exp)
    trial_indices = np.load('./stats/%s_experiment/condition_subsample_indices.npz' % exp)
    tmp = {}
    for condition in config['conditions']:
        tmp[condition] = trial_indices[condition]
    trial_indices = tmp

    permutation_info = {}

    # loop through condition comparisons
    comparisons = [['Open', 'Closed'], ['Open', 'Brain'], ['Brain', 'Closed']]
    for comp in comparisons:

        print('Computing Permutation Distribution for Condition Comparison: %s-%s' %(comp[0], comp[1]))

        # collect all power for the relevant conditions
        all_conditions_power = {}
        for condition in comp:
            power, chs, times, freqs = load_all_data('saline', condition)
            all_conditions_power[condition] = power

        # get the permutation index
        permutation_ix = permutation_indices['%s_%s' % (comp[0],
                                                        comp[1])]

        # compute the base difference
        base_diffs = compute_permutation_sample(-1, all_conditions_power,
                                                trial_indices, permutation_ix,
                                                times, freqs, chs, config, comp)
        for i, band in enumerate(['alpha', 'beta']):
            permutation_info['%s_diff' % band] = base_diffs[i]

        num_permutations = permutation_indices['num_permutations']
        perm_diffs = Parallel(config['n_jobs'])(delayed(compute_permutation_wrapper)(ix) for ix in range(num_permutations))

        permutation_info['alpha_perm_dist'] = [diff[0] for diff in perm_diffs]
        permutation_info['beta_perm_dist'] = [diff[1] for diff in perm_diffs]

        # compute p-values
        permutation_info['alpha_p_value'] = compute_permutation_p_value(permutation_info['alpha_diff'],
                                                                        permutation_info['alpha_perm_dist'])
        permutation_info['beta_p_value'] = compute_permutation_p_value(permutation_info['beta_diff'],
                                                                        permutation_info['beta_perm_dist'])

        # save the permutation information
        np.savez_compressed('./stats/%s_experiment/%s-%s_%s_permutation_info.npz' %(exp, comp[0], comp[1], exp),
                            alpha_dist = permutation_info['alpha_perm_dist'],
                            beta_dist = permutation_info['beta_perm_dist'],
                            alpha_diff = permutation_info['alpha_diff'],
                            beta_diff=permutation_info['beta_diff'],
                            num_permutations=num_permutations,
                            alpha_p_value = permutation_info['alpha_p_value'],
                            beta_p_value = permutation_info['beta_p_value'])


def compute_permutation_wrapper(ix):
    return compute_permutation_sample(ix, all_conditions_power,
                                      trial_indices,
                                      permutation_ix,
                                      times, freqs, chs, config, comp)


# Plotting Functions


def plot_before_during_after_spectra(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style='white', font_scale=config['font_scale'],
            rc={"lines.linewidth": config['linewidth']})

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    for i, time_period in enumerate(['Before', 'During', 'After']):

        ax = axs[i]

        for j, condition in enumerate(config['conditions']):

            power, chs, times, freqs = load_all_data(exp, condition)

            power = reduce_array_power(power, chs, config['bad_chs'], axis=1)

            power = reduce_toi_power(power, times, config[time_period],
                                     axis=-1)

            bootstrap_dist = simple_bootstrap(power, config['num_bootstraps'],
                                              axis=0)

            # reduce over trials
            power = power.mean(axis=0)
            bootstrap_dist = bootstrap_dist.mean(axis=1)

            # normalize spectra
            power /= power.sum()
            bootstrap_dist /= bootstrap_dist.sum(axis=-1)[:, np.newaxis]

            # extract bootstrap standard error
            bootstrap_std_err = bootstrap_dist.std(axis=0)

            # plot the spectra with standard error shading
            ax.plot(freqs, power, color=config['colors'][j])
            ax.fill_between(freqs, power - bootstrap_std_err,
                            power + bootstrap_std_err,
                            color=config['colors'][j],
                            alpha=0.5, label='_nolegend_')

        ax.set_title('%s Stimulation Power' % time_period)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Normalized Power')
        ax.set_ylim((0, 0.5))

    axs[-1].legend(config['conditions'])
    plt.tight_layout()
    sns.despine()

    return fig

def plot_bootstrap_distributions(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['linewidth']})

    (fig, axs) = plt.subplots(2, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, condition in enumerate(config['conditions']):

        bootstrap_info = np.load("./stats/saline_experiment/%s_bootstrap_info.npz" % condition)

        for j, band in enumerate(['alpha', 'beta']):
            dist = bootstrap_info['%s_dist' % band]
            power = bootstrap_info['%s' % band]
            p = bootstrap_info['%s_p' % band]

            # reduce to toi power
            times = bootstrap_info['times']
            dist = np.sort(reduce_toi_power(dist, times, config['toi'],
                                            axis=-1))
            power = reduce_toi_power(power, times, config['toi'], axis=-1)

            # extract 95% confidence interval
            lower_ix = int(len(dist) * .025)
            upper_ix = int(len(dist) * .975)
            ci = [dist[lower_ix], dist[upper_ix]]

            # plot bootstrap distribution with actual value and ci marked
            ax = axs[j, i]
            sns.distplot(dist, ax=ax, color=config['colors'][i])
            ax.axvline(ci[0], color='k')
            ax.axvline(ci[1], color='k')
            ax.axvline(power, color=config['colors'][i], linewidth=2)
            ax.set_title("%s %s Bootstrap Distribution \n Uncorrected p = %.3f" % (condition, band, p))

    plt.tight_layout()
    sns.despine()
    return fig


def plot_permutation_distributions(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    comparisons = ["Open-Closed", "Open-Brain", "Brain-Closed"]
    ps = []

    (fig, axs) = plt.subplots(2, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, comp in enumerate(comparisons):
        perm_info = np.load("./stats/%s_experiment/%s_%s_permutation_info.npz" % (exp, comp, exp))

        # plot permutation distribution
        ax = axs[0, i]
        sns.distplot(perm_info['alpha_dist'], ax=ax)
        ax.axvline(perm_info['alpha_diff'], color=config['colors'][1])
        ax.set_title("%s Alpha Power \n Uncorrected p = %.3f" %(comp, perm_info['alpha_p_value']))

        ax = axs[1, i]
        sns.distplot(perm_info['beta_dist'], ax=ax)
        ax.axvline(perm_info['beta_diff'], color=config['colors'][1])
        ax.set_title("%s Beta Power \n Uncorrected p = %.3f" %(comp, perm_info['beta_p_value']))

        ps.append(perm_info['alpha_p_value'])
        ps.append(perm_info['beta_p_value'])

    plt.tight_layout()
    sns.despine()
plt.savefig("./plots/saline_experiment/saline_condition_permutation_dists.png")
plt.show()


def plot_condition_toi_comparison(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    fig, axs = plt.subplots(1, 2, figsize = (22, 8))

    for bar_ix, color, c in zip(range(1, len(config['conditions']) + 1),
                                config['colors'],
                                config['conditions']):

        bootstrap_info = np.load("./stats/%s_experiment/%s_bootstrap_info.npz" % (exp, c))
        times = bootstrap_info['times']

        for i, band in enumerate(['alpha', 'beta']):
            dist = bootstrap_info['%s_dist' % band]
            power = bootstrap_info['%s' % band]

            times = bootstrap_info['times']
            toi_mask = np.where(np.logical_and(times >= config['toi'][0],
                                               times <= config['toi'][1]))[0]
            dist = np.sort(dist[:, toi_mask].mean(axis=-1))
            power = power[toi_mask].mean()

            lower_ix = int(len(dist) * .025)
            upper_ix = int(len(dist) * .975)

            ci = [dist[lower_ix], dist[upper_ix]]

            axs[i].bar(bar_ix, power, color=color)
            axs[i].plot([bar_ix + .4, bar_ix + .4], ci, color='k',
                        label='_nolegend_')

            axs[i].set_title("%s Power" % band.capitalize())

            axs[i].set_xlim((.8, 4))
            axs[i].set_xticks(())

    axs[1].legend(config['conditions'], loc=8)
    axs[0].axhline(0, color='k')
    axs[1].axhline(0, color='k')
    axs[0].set_ylabel("dB Change From Baseline")

    # Statistical Annotations
    # x1, x2 = 1.4, 2.38
    # y, h, col = .4, .1, 'k'
    # axs[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
    # axs[0].text((x1+x2)*.5, y+h, "p = .041", ha='center', va='bottom', color=col)

    # x1, x2 = 2.42, 3.4
    # y, h, col = .4, .1, 'k'
    # axs[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
    # axs[0].text((x1+x2)*.5, y+h, "p = .016", ha='center', va='bottom', color=col)

    sns.despine()
    return fig
    f.savefig("./plots/saline_experiment/saline_post-stim_toi_condition_comparison.png")
    plt.show()


def plot_condition_band_comparison(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    fig, axs = plt.subplots(1, 2, figsize = (22, 8))

    for color, c in zip(config['colors'], config['conditions']):

        bootstrap_info = np.load("./stats/%s_experiment/%s_bootstrap_info.npz" % (exp, c))

        times = bootstrap_info['times']
        pre_mask = np.logical_and(times >= -10, times <= -.5)
        post_mask = np.logical_and(times >= 10.5, times <= 20)
        time_mask = np.where(np.logical_or(pre_mask, post_mask))[0]
        times = times[time_mask]
        times[times >= 10] -= 10

        for i, band in enumerate(['alpha', 'beta']):

            dist = bootstrap_info['%s_dist' % band]
            power = bootstrap_info['%s' % band]
            dist = dist[:, time_mask]
            power = power[time_mask]

            axs[i].plot(times, power, color=color)
            axs[i].fill_between(times, power - dist.std(axis=0),
                                power + dist.std(axis=0),
                                facecolor=color, alpha=0.2, edgecolor='none')

            axs[i].set_title("%s Power" % band.capitalize())

            axs[i].set_xlabel("Time (s)")

            axs[i].set_ylim((-5, 5))
            axs[i].set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
            axs[i].set_xticklabels([-5, -4, -3, -2, -1, 'Stim', 1, 2, 3, 4, 5])
            axs[i].set_xlim((-10, 10))

            sns.despine()

    for i in np.arange(-.5, .5, .01):
        axs[0].axvline(i, color='k', alpha=0.8)
        axs[1].axvline(i, color='k', alpha=0.8)
    axs[0].legend(config['conditions'])
    axs[0].set_ylabel("dB Change From Baseline")

    sns.despine()
    return fig
