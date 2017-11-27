import numpy as np
import json
from data_processing import load_power_data, reduce_toi_power
from data_processing import reduce_array_power, reduce_band_power
from data_processing import baseline_normalize
from joblib import Parallel, delayed
from scipy.stats import ttest_ind


# Bootstrap Functions


def pre_compute_bootstrap_indices(exp):
    """ Pre-computes bootstrap re-sampled indices to estimate error of
        post-stimulation band power.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        None. Instead, writes out re-sampled indices for each condition
        to compressed numpy file.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    np.random.seed(config['random_seed'])

    bootstrap_indices = {}
    for ss, c in zip(config['%s_sample_sizes' % exp], config['conditions']):
        bootstrap_indices[c] = np.random.choice(ss,
                                                size=(config['num_bootstraps'],
                                                      ss),
                                                replace=True)
    f = '../data/stats/%s_experiment/condition_bootstrap_indices.npz' % exp
    np.savez_compressed(f, Open=bootstrap_indices['Open'],
                        Closed=bootstrap_indices['Closed'],
                        Brain=bootstrap_indices['Brain'],
                        num_samples=config['num_bootstraps'])


def compute_bootstrap_p_value(power, bootstrap_dist, times, toi):
    """ Computes a bootstrap p-value to test the null hypothesis that
        post stimuation band power within a time period of interest is equal
        to zero.

        Checks to see how unlikely the given post-stimulation band power
        within a time period of interest is, given a bootstrapped null
        distribution that we shift to center around 0. It does this by
        calculating the % of bootstrapped values that are more extreme than
        the calculated value.

        Args:
            power: The time series consisting of the non-bootstrapped band
                power.
            bootstrap_dist: The bootstrap distribution of band power.
            times: List of time labels.
            toi: Tuple containing the limits for the time period of interest.

        Returns:
            Returns the p-value (float) corresponding to the percentage of
            bootstrapped toi values that were more extreme than the estimated
            toi band power value.
    """
    # reduce to toi
    bootstrap_dist = reduce_toi_power(bootstrap_dist, times, toi, axis=-1)
    power = reduce_toi_power(power, times, toi, axis=-1)

    # sort by toi value
    bootstrap_dist = np.sort(bootstrap_dist)

    # center the distribution to make it a "null distribution"
    # assumes symmetry of the distribution
    bootstrap_dist = bootstrap_dist - power

    # compute the p-value as the percentage of bootstrap values larger
    # in absolute value than the
    p_num = np.sum(np.abs(bootstrap_dist) >= np.abs(power)) + 1.
    p_denom = len(bootstrap_dist) + 1.

    return p_num / p_denom


def compute_bootstrap_sample(bootstrap_ix, power, times, freqs, chs, config,
                             exp):
    """ Helper function to compute the bootstrapped band power for a
    particular re-sampled index.

    This function takes in the tfr power data, re-samples the trials in the
    data according to the given re-sampled index, and then computes
    baseline normalized band power averaged across the first recording array.

    Args:
        bootstrap_ix: A pre-computed trial re-sampling index.
        power: The raw tfr power to compute the re-sampled band power on.
        times: List of time labels.
        freqs: List of frequency labels.
        chs: List of channel names.
        config: Dictionary containing experiment wide configuration info. In
            this case it contains the baseline period to normalize to, bad chs
            to ignore, and the frequency ranges for alpha and beta band.

    Returns:
        The re-sampled band power time series for the alpha and beta bands
        in a length two list.
    """

    # resample the data
    power = power[bootstrap_ix, :, :, :].squeeze()

    # baseline normalize
    power = baseline_normalize(power, config['baseline'], times)

    # reduce over array
    power = reduce_array_power(power, chs, config['%s_bad_chs' % exp], '1',
                               axis=0)

    # reduce over band
    output = []
    for band in ['alpha', 'beta']:
        output.append(reduce_band_power(power, freqs, config[band], axis=0))

    return output


def compute_bootstrap_wrapper(ix):
    """ Simple wrapper function to facilitate parallelization of the
        bootstrap sampling.

        This function allows a single parameter function to be used for
        parallelization. It simply calls compute_bootstrap_sample making
        use of several arguments that were made global in
        compute_bootstrap_distributions.

        Args:
            ix: The index of the re-sampled indices to use.
    """

    return compute_bootstrap_sample(bootstrap_cond_ix[ix], power,
                                    times, freqs, chs, config, exper)


def compute_bootstrap_distribution(exp):
    """ Computes bootstrap distributions for alpha and beta band power for all
    three stimulation conditions.

    For each condition, it loads that condition's raw tfr and pre-computed
    bootstrap sampled indices. It then runs through each re-sampled index
    and computes the re-sampled band power to create a bootstrap distribution.
    Finally, it computes a bootstrap p-value testing for post-stimulation
    toi power differences from 0.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        None. It saves all of the bootstrap information, including the
        band power estimates, the band power bootstrap distributions, and
        the post-stimulation toi bootstrap p-values into a compressed
        numpy file.
    """

    global power, times, freqs, chs, bootstrap_cond_ix, config, exper
    exper = exp

    # load in configurations
    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    # load in pre-computes bootstrap re-sample indices
    f = '../data/stats/%s_experiment/condition_bootstrap_indices.npz' % exp
    bootstrap_indices = np.load(f)
    num_bootstrap_samples = bootstrap_indices['num_samples']

    for condition in config['conditions']:

        print('Computing Bootstrap Distribution for Condition: %s' % condition)

        power, chs, times, freqs = load_power_data(exp, condition)

        # compute the base band power
        base_ix = np.arange(power.shape[0])
        alpha_power, beta_power = compute_bootstrap_sample(base_ix, power,
                                                           times, freqs, chs,
                                                           config, exp)

        # loop through all bootstrap samples in parallel
        bootstrap_cond_ix = bootstrap_indices[condition]
        par = Parallel(n_jobs=config['n_jobs'])
        bootstrap_samples = par(delayed(compute_bootstrap_wrapper)(ix)
                                for ix in range(num_bootstrap_samples))

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
        f = '../data/stats/%s_experiment/%s_bootstrap_info.npz' % (exp, condition)
        np.savez_compressed(f, alpha=alpha_power, beta=beta_power,
                            alpha_dist=alpha_bootstrap_samples,
                            beta_dist=beta_bootstrap_samples,
                            alpha_p=alpha_p, beta_p=beta_p, times=times)


def simple_bootstrap(data, axis):
    """ Computes a bootstrap re-sampled distribution for data on the given
        axis.

        Args:
            data: The data to compute the bootstrap distribution for.
            axis: The axis along which to resample.

        Returns:
            Returns the bootstrap distribution for the re-sampled data.
    """
    # load in configurations
    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    np.random.seed(config['random_seed'])

    bootstrap_indices = np.random.choice(data.shape[axis],
                                         size=(config['num_bootstraps'],
                                               data.shape[axis]),
                                         replace=True)

    bootstrap_samples = np.zeros([config['num_bootstraps']] +
                                 [dim for dim in data.shape])
    for i in range(config['num_bootstraps']):
        bootstrap_samples[i, :] = data[bootstrap_indices[i, :], :]

    return bootstrap_samples


# Permutation Testing Functions


def pre_compute_subsample_indices(exp):
    """ Pre-computes subsample indices to equalize trial counts between
    conditions to faciliate permuation testing.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        None. Saves out the sub-sampled indices for each condition to file.
    """
    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    np.random.seed(config['random_seed'])

    trial_indices = {}

    sample_sizes = config['%s_sample_sizes' % exp]
    oss = max(sample_sizes)
    dss = min(sample_sizes)

    # only subsample for open and closed
    for c in config['conditions'][:2]:

        trial_indices[c] = np.zeros((config['num_permutations'], dss),
                                    dtype=np.int32)
        for i in range(config['num_permutations']):
            trial_indices[c][i, :] = np.random.choice(oss, size=dss,
                                                      replace=False)

    f = '../data/stats/%s_experiment/condition_subsample_indices.npz'
    np.savez_compressed(f % exp, Closed=trial_indices['Closed'],
                        Open=trial_indices['Open'])


def pre_compute_permutation_indices(exp):
    """ Pre-computes permutation indices to permute trial condition labels
    between two stimulation conditions of data.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        None. Saves out the pemurtation indices for each condition pair to
        file.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    tests = ['Open-Closed', 'Open-Brain', 'Brain-Closed']

    np.random.seed(config['random_seed'])

    permutation_indices = {}
    b_ss = min(config['%s_sample_sizes' % exp])
    oc_ss = max(config['%s_sample_sizes' % exp])
    sses = [oc_ss, b_ss, b_ss]

    for ss, t in zip(sses, tests):
        permutations = np.zeros((config['num_permutations'], ss * 2),
                                dtype=np.int32)
        ix = np.arange(ss * 2)
        for i in range(config['num_permutations']):
            np.random.shuffle(ix)
            permutations[i, :] = ix
        permutation_indices[t] = permutations

    # Don't save out main indices due to
    if exp != 'main':
        f = '../data/stats/%s_experiment/condition_permutation_indices.npz'
        np.savez_compressed(f % exp,
                            Open_Closed=permutation_indices['Open-Closed'],
                            Open_Brain=permutation_indices['Open-Brain'],
                            Brain_Closed=permutation_indices['Brain-Closed'],
                            num_permutations=config['num_permutations'])


def compute_permutation_p_value(base_power, permutation_dist):
    """ Computes a permutation test p-value.

    P-value computed as the percentage of values in the permutation
    distribution more extreme than the actual test statistic (difference)
    of two condition band power toi averages.

    Args:
        base_power: The non-permuted difference in band toi power between
            conditions.
        permutation_dist: The permutation distribution of band toi power
            differences.

    Returns:
        Permutation p-value (float).

    """
    num = np.sum(np.abs(np.array(permutation_dist)) >=
                 np.abs(base_power)) + 1.
    denom = len(permutation_dist) + 1.
    return num / denom


def compute_permutation_sample(perm_num, all_conditions_power, trial_indices,
                               permutation_indices, times, freqs, chs, config,
                               comp, exp):
    """ Helper function to compute the permuted toi band power difference for
    a particular permutation of trials between two conditions.

    This function takes in the tfr power data for two conditions, permutes
    the trial membership between the two conditions according to the given
    permutation index, and then computes the baseline-normalized band power
    averaged across the first recording array and a post-stimulation time
    period of interest for each condition and returns their difference.

    Args:
        perm_num: The permutation number used to index a particular
            permutation index and sub-sample index.
        all_conditions_power: Dictionary containing the tfr power data for
            each condition being tested.
        trial_indices: The pre-computed sub-sampling indices.
        permutation_indices: The pre-computed permutation indices.
        times: List of time labels.
        freqs: List of frequency labels.
        chs: List of channel names.
        config: Dictionary containing experiment wide configuration info. In
            this case it contains the baseline period to normalize to, bad chs
            to ignore, time period to average over, and the frequency ranges
            for alpha and beta band.
        comp: List of the two conditions to compare.

    Returns:
        List of two numbers representing the permuted difference between
        the two conditions for the alpha and beta bands.
    """

    # collect power across conditions into single array
    # we downsample to match trial sizes
    power = []
    for c in comp:
        if perm_num != -1 and c != 'Brain' and 'Brain' in comp:
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
    power[0] = reduce_array_power(power[0], chs, config['%s_bad_chs' % exp],
                                  '1', axis=0)
    power[1] = reduce_array_power(power[1], chs, config['%s_bad_chs' % exp],
                                  '1', axis=0)

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


# def compute_permutation_wrapper(ix):
#     """ Simple wrapper function to facilitate parallelization of the
#     permutations.

#     This function allows a single parameter function to be used for
#     parallelization. It simply calls compute_permutation_sample making
#     use of several arguments that were made global in
#     compute_permutation_distributions.

#     Args:
#         ix: The index of the re-sampled indices to use.
#     """

#     return compute_permutation_sample(ix, all_conditions_power,
#                                       trial_indices,
#                                       permutation_ix,
#                                       times, freqs, chs, config, comp, exper)


def compute_permutation_distributions(exp):
    """ Computes permutation distributions for alpha and beta band power
    for all three pairs stimulation conditions differences.

    For each condition pair, it loads those condition's raw tfr power data.
    It then runs through each sub-sample index and permutation index to
    equalize trial counts and then permute trial membership between the two
    conditions in order to calculate the difference between the mean
    post-stimulation band toi power creating a permutation distribution.
    Finally, it computes a permutation p-value testing for post-stimulation
    toi power differences between conditions.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        None. It saves all of the permutation information, including the
        band power toi difference estimates, the permutation distributions,
        and the post-stimulation condition difference p-values into a
        compressed numpy file.
    """

    exper = exp

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    # load pre-sampled indices
    f = '../data/stats/%s_experiment/condition_permutation_indices.npz' % exp
    permutation_indices = np.load(f)
    f = '../data/stats/%s_experiment/condition_subsample_indices.npz' % exp
    trial_indices = np.load(f)
    tmp = {}
    for condition in config['conditions']:
        tmp[condition] = trial_indices[condition]
    trial_indices = tmp

    permutation_info = {}

    # loop through condition comparisons
    comparisons = [['Open', 'Closed'], ['Open', 'Brain'], ['Brain', 'Closed']]
    for comp in comparisons:

        print('Computing Permutation Distribution for Condition ' +
              'Comparison: %s-%s' % (comp[0], comp[1]))

        # collect all power for the relevant conditions
        all_conditions_power = {}
        for condition in comp:
            power, chs, times, freqs = load_power_data(exp, condition)
            all_conditions_power[condition] = power

        # get the permutation index
        permutation_ix = permutation_indices['%s_%s' % (comp[0],
                                                        comp[1])]

        # compute the base difference
        base_diffs = compute_permutation_sample(-1, all_conditions_power,
                                                trial_indices, permutation_ix,
                                                times, freqs, chs, config,
                                                comp, exp)
        for i, band in enumerate(['alpha', 'beta']):
            permutation_info['%s_diff' % band] = base_diffs[i]

        num_permutations = permutation_indices['num_permutations']
        perm_diffs = []
        for ix in range(num_permutations):
            perm_diffs.append(compute_permutation_sample(ix,
                                                         all_conditions_power,
                                                         trial_indices,
                                                         permutation_ix,
                                                         times, freqs, chs,
                                                         config, comp, exper))

        permutation_info['alpha_perm_dist'] = [diff[0] for diff in perm_diffs]
        permutation_info['beta_perm_dist'] = [diff[1] for diff in perm_diffs]

        # compute p-values
        tmp = compute_permutation_p_value(permutation_info['alpha_diff'],
                                          permutation_info['alpha_perm_dist'])
        permutation_info['alpha_p_value'] = tmp

        tmp = compute_permutation_p_value(permutation_info['beta_diff'],
                                          permutation_info['beta_perm_dist'])
        permutation_info['beta_p_value'] = tmp

        # save the permutation information
        f = '../data/stats/%s_experiment/%s-%s_%s_permutation_info.npz'
        np.savez_compressed(f % (exp, comp[0], comp[1], exp),
                            alpha_dist=permutation_info['alpha_perm_dist'],
                            beta_dist=permutation_info['beta_perm_dist'],
                            alpha_diff=permutation_info['alpha_diff'],
                            beta_diff=permutation_info['beta_diff'],
                            num_permutations=num_permutations,
                            alpha_p_value=permutation_info['alpha_p_value'],
                            beta_p_value=permutation_info['beta_p_value'])


def compute_array_permutation_distribution(exp):
    """ Computes permutation distributions for alpha and beta band power
    difference between the two recording arrays for all three conditions.

    For each condition, it loads those condition's raw tfr power data. It
    then baseline normalizes the power and reduces to band power and averages
    over a post-stimulation time of interest. Then it permutes recording
    array membership to generate a permutation distribution of differences
    between recording array averages. Finally, it computes a permutation
    p-value testing for post-stimulation toi power differences between
    recording arrays for each condition.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        None. It saves all of the permutation information, including the
        band power toi difference estimates, the permutation distributions,
        and the post-stimulation array difference p-values into a
        compressed numpy file.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    for condition in config['conditions']:
        print(condition)

        np.random.seed(config['random_seed'])

        # load all data for condition
        power, chs, times, freqs = load_power_data(exp, condition)

        # baseline normalize and reduce to time of interest
        power = baseline_normalize(power, config['baseline'], times)
        power = reduce_toi_power(power, times, config['toi'], axis=-1)

        perm_info = {}
        for band in ['alpha', 'beta']:
            perm_info['%s_perm_dist' % band] = []

        # build the permutation distribution
        for i in range(config['num_permutations'] + 1):
            if i > 0:
                # shuffle channel array membership
                np.random.shuffle(chs)

            # select out array 1 and array 2 channels
            arr1_ix = [ix for ix in np.arange(len(chs))
                       if 'elec1' in chs[ix] and
                       chs[ix] not in config['%s_bad_chs' % exp]]
            arr2_ix = [ix for ix in np.arange(len(chs)) if 'elec2' in chs[ix]]

            for band in ['alpha', 'beta']:

                # reduce to desired band
                band_power = reduce_band_power(power, freqs, config[band],
                                               axis=-1)

                if i == 0:
                    tmp = '%s_diff' % band
                    perm_info[tmp] = ttest_ind(band_power[arr1_ix],
                                               band_power[arr2_ix])[0]
                else:
                    tmp = '%s_perm_dist' % band
                    perm_info[tmp].append(ttest_ind(band_power[arr1_ix],
                                                    band_power[arr2_ix])[0])

        for band in ['alpha', 'beta']:
            # compute the p-value
            tmp1 = '%s_p_value' % band
            tmp2 = '%s_diff' % band
            tmp3 = '%s_perm_dist' % band
            perm_info[tmp1] = compute_permutation_p_value(perm_info[tmp2],
                                                          perm_info[tmp3])

        # save permutation info to file
        f = '../data/stats/%s_experiment/' % exp + \
            '%s_array_permutation_info.npz' % condition
        np.savez_compressed(f, alpha_dist=perm_info['alpha_perm_dist'],
                            beta_dist=perm_info['beta_perm_dist'],
                            alpha_diff=perm_info['alpha_diff'],
                            beta_diff=perm_info['beta_diff'],
                            alpha_p_value=perm_info['alpha_p_value'],
                            beta_p_value=perm_info['beta_p_value'])

    print('Done!')
