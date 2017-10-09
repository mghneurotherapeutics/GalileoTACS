import numpy as np
import json
from data_processing import load_all_data, reduce_toi_power
from data_processing import reduce_array_power, reduce_band_power
from data_processing import baseline_normalize
from joblib import Parallel, delayed

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

    global all_conditions_power, times, freqs, chs, trial_indices
    global permutation_ix, comp, config

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    # load pre-sampled indices
    f = './stats/%s_experiment/condition_permutation_indices.npz' % exp
    permutation_indices = np.load(f)
    f = './stats/%s_experiment/condition_subsample_indices.npz' % exp
    trial_indices = np.load(f)
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
