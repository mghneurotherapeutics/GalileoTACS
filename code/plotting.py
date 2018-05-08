import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
from data_processing import load_power_data, reduce_toi_power
from data_processing import baseline_normalize
from data_processing import reduce_band_power, reduce_array_power
from statistics import simple_bootstrap


# main effects

# condition comparisons

def plot_condition_band_comparison(exp):
    """ Plots the pre- and post-stimulation alpha and beta time series
    for all three conditions.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 1 x 2 matplotlib figure. Each subplot contains alpha and beta band
        time series for all three conditions with bootstrap standard error
        shading. The stimulation period is ignored and centered at 0 with a
        +- 0.5 blacked out period representing stimulation edge artifact.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    fig, axs = plt.subplots(1, 2, figsize=(22, 8))

    for color, c in zip(config['colors'], config['conditions']):

        f = '../data/stats/%s_experiment/%s_bootstrap_info.npz'
        bootstrap_info = np.load(f % (exp, c))

        # remove the stimulation period from the time labels
        times = bootstrap_info['times']
        pre_mask = np.logical_and(times >= -config['tfr_epoch_width'],
                                  times <= -.5)
        post_mask = np.logical_and(times >= 10.5,
                                   times <= 10 + config['tfr_epoch_width'])
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

            axs[i].set_ylim((-1, 1))

            xlim = 3
            xticks = list(np.arange(-xlim, xlim + 1))
            xticklabels = ['Stim' if x == 0 else x for x in xticks]
            axs[i].set_xticks(xticks)
            axs[i].set_xticklabels(xticklabels)
            axs[i].set_xlim((-xlim, xlim))

            sns.despine()

    # blackout the stimulation period
    for i in np.arange(-.5, .5, .01):
        axs[0].axvline(i, color='k', alpha=0.8)
        axs[1].axvline(i, color='k', alpha=0.8)

    axs[0].legend(config['conditions'])
    axs[0].set_ylabel("dB Change From Baseline")

    sns.despine()
    return fig


def plot_condition_toi_comparison(exp):
    """ Plots the pre- and post-stimulation alpha and beta time of interest
    averages for all three conditions.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 1 x 2 matplotlib figure. Each subplot contains alpha and beta band
        toi average barplots for all three conditions with bootstrap 95% CI
        bars and significance marking between conditions based on
        permutation testing.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    fig, axs = plt.subplots(1, 2, figsize=(22, 8))

    for bar_ix, color, c in zip(range(1, len(config['conditions']) + 1),
                                config['colors'],
                                config['conditions']):

        f = '../data/stats/%s_experiment/%s_bootstrap_info.npz'
        bootstrap_info = np.load(f % (exp, c))
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
            axs[i].plot([bar_ix, bar_ix], ci, color='k',
                        label='_nolegend_')

            axs[i].set_title("%s Power" % band.capitalize())

            axs[i].set_xlim((.4, 3.6))
            axs[i].set_xticks(())
            axs[i].set_ylim((-.7, .7))

    axs[1].legend(config['conditions'], loc=8)
    axs[0].axhline(0, color='k')
    axs[1].axhline(0, color='k')
    axs[0].set_ylabel("dB Change From Baseline")

    # Statistical Annotations
    if exp == 'main':
        x1, x2 = 1, 1.98
        y, h, col = .4, .1, 'k'
        axs[0].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.5, c=col)
        axs[0].text((x1 + x2) * .5, y + h, "p = .041", ha='center',
                    va='bottom', color=col)

        x1, x2 = 2.02, 3.0
        y, h, col = .4, .1, 'k'
        axs[0].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.5, c=col)
        axs[0].text((x1 + x2) * .5, y + h, "p = .016", ha='center',
                    va='bottom', color=col)

    sns.despine()
    return fig


# array comparisons


def plot_array_band_comparison(exp):
    """ Plots the pre- and post-stimulation alpha and beta time series
    for all three conditions compared between recording arrays.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 2 x 3 matplotlib figure (frequency band x condition). Each subplot
        contains array1 and array2 time series for a particular condition and
        frequency band with bootstrap standard error shading. The stimulation
        period is ignored and centered at 0 with a +- 0.5 blacked out period
        representing stimulation edge artifact.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    # plotting initialization
    sns.set(style='white', font_scale=config['font_scale'],
            rc={"lines.linewidth": config['linewidth']})
    fig, axs = plt.subplots(2, 3, figsize=(22, 10))
    plt.subplots_adjust(hspace=.3)

    window = 3
    xticks = np.arange(-window, window + 1)
    xticklabels = ['Stim' if x == 0 else x for x in xticks]
    ls = ['-', '--']

    # hack the legend to be color agnostic
    axs[0, 2].axvline(-3, color='k')
    axs[0, 2].axvline(-3, color='k', linestyle='--')
    axs[0, 2].legend(['Array 1', 'Array 2'])

    for i, c in enumerate(config['conditions']):

        power, chs, times, freqs = load_power_data(exp, c)

        power = baseline_normalize(power, config['baseline'], times)

        # select out pre and post stimulation
        # collapse stimulation into 0 and make pre and post stimulation times
        # relative to this 0 (so no longer 10 + for post stimulation)
        pre_mask = np.logical_and(times >= -5, times <= -.5)
        post_mask = np.logical_and(times >= 10.5, times <= 15)
        time_mask = np.where(np.logical_or(pre_mask, post_mask))[0]
        times = times[time_mask]
        power = power[:, :, time_mask]
        times[times >= 10] -= 10

        # array indices
        arr1_ix = [ix for ix in np.arange(len(chs)) if 'elec1' in chs[ix] and
                   chs[ix] not in config['%s_bad_chs' % exp]]
        arr2_ix = [ix for ix in np.arange(len(chs)) if 'elec2' in chs[ix]]

        for j, band in enumerate(['alpha', 'beta']):

            band_power = reduce_band_power(power, freqs, config[band], axis=1)

            for k, arr in enumerate([arr1_ix, arr2_ix]):
                arr_power = band_power[arr, :].mean(axis=0)
                arr_stderr = band_power[arr, :].std(axis=0) / \
                    np.sqrt(len(arr))

                axs[j, i].plot(times, arr_power, color=config['colors'][i],
                               linestyle=ls[k])
                axs[j, i].fill_between(times, arr_power - arr_stderr,
                                       arr_power + arr_stderr,
                                       facecolor=config['colors'][i],
                                       alpha=0.2, edgecolor='none',
                                       label='_nolegend_')

            # pretty axis
            axs[j, i].set_title('%s %s Power' % (c, band.capitalize()))
            axs[j, i].set_xlim((-window, window))
            axs[j, i].set_xticks(xticks)
            axs[j, i].set_xticklabels(xticklabels)
            axs[j, i].set_ylim((-1, 1))
            if i == 0:
                axs[j, i].set_ylabel('dB Change From Baseline')
            if j == 1:
                axs[j, i].set_xlabel('Time (s)')

            # add blackout for stim period
            for x in np.arange(-.5, .5, .01):
                axs[j, i].axvline(x, color='k', alpha=0.8, label='_nolegend_')
                axs[j, i].axvline(x, color='k', alpha=0.8, label='_nolegend_')

    plt.tight_layout()
    sns.despine()

    return fig


def plot_array_toi_comparison(exp):
    """ Plots the pre- and post-stimulation alpha and beta time of interest
    averages for all three conditions comparing the two arrays.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 1 x 2 matplotlib figure. Each subplot contains alpha and beta band
        toi average barplots for all three conditions split by recording array
        with bootstrap standard error bars and significance marking between
        array averages based on permutation testing.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    # plotting initialization
    sns.set(style='white', font_scale=config['font_scale'],
            rc={"lines.linewidth": config['linewidth']})
    fig, axs = plt.subplots(1, 2, figsize=(22, 10))
    plt.subplots_adjust(hspace=.3)

    stat_ys = [-.3, .15, -.4, -.2, .15, -.35]
    stat_hmults = [3, 1.5, 3, 3, 1.5, 3]
    stat_hs = [-.03, .02, -.03, -.03, .02, -.03]

    for i, c in enumerate(config['conditions']):

        f = '../data/stats/%s_experiment/%s_array_permutation_info.npz'
        perm_info = np.load(f % (exp, c))

        power, chs, times, freqs = load_power_data(exp, c)

        power = baseline_normalize(power, config['baseline'], times)

        # array indices
        arr1_ix = [ix for ix in np.arange(len(chs)) if 'elec1' in chs[ix] and
                   chs[ix] not in config['%s_bad_chs' % exp]]
        arr2_ix = [ix for ix in np.arange(len(chs)) if 'elec2' in chs[ix]]

        for j, band in enumerate(['alpha', 'beta']):

            band_power = reduce_band_power(power, freqs, config[band], axis=1)
            toi_power = reduce_toi_power(band_power, times, config['toi'],
                                         axis=-1)
            

            for k, arr in enumerate([arr1_ix, arr2_ix]):
                arr_power = toi_power[arr].mean(axis=0)
                
                dist = np.sort(simple_bootstrap(toi_power[arr][:, np.newaxis], axis=0).squeeze().mean(axis=0))
                lower_ix = int(len(dist) * .025)
                upper_ix = int(len(dist) * .975)
                ci = [dist[lower_ix], dist[upper_ix]]

                bar_tick = i * 2 + k * .8

                if k == 0:
                    axs[j].bar(bar_tick, arr_power, color=config['colors'][i])
                    axs[j].plot([bar_tick + .4, bar_tick + .4], ci, color='k',
                                label='_nolegend_')
                else:
                    axs[j].bar(bar_tick, arr_power, facecolor='none',
                               edgecolor=config['colors'][i], linewidth=4,
                               hatch='/')
                    axs[j].plot([bar_tick + .4, bar_tick + .4], ci, color='k',
                                label='_nolegend_')

            # pretty axis
            axs[j].set_title('%s Power' % band.capitalize(), y=1.05)
            axs[j].set_xticks([x + .8 for x in [0, 2, 4]])
            axs[j].set_xticklabels(config['conditions'])
            axs[j].set_ylim((-.7, .7))
            axs[j].set_xlim((-.6, 6.4))
            axs[j].set_ylabel('dB Change From Baseline')
            axs[j].axhline(0, color='k', label='_nolegend_')

            # statistical annotation
            p = perm_info['%s_p_value' % band]
            if p < .0002:
                p = 'p < .0002'
            else:
                p = 'p = %.04f' % p
            x1, x2 = i * 2 + .4, i * 2 + 1.2
            y = stat_ys[j * 3 + i]
            hmult = stat_hmults[j * 3 + i]
            h = stat_hs[j * 3 + i]
            axs[j].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.5, c='k',
                        label='_nolegend_')
            axs[j].text((x1 + x2) * .5, y + hmult * h, p, ha='center',
                        va='bottom', color='k', size=22)

    # set legend
    axs[1].legend(["Array 1", "Array 2"])
    leg = axs[1].get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_edgecolor('black')

    plt.tight_layout()
    sns.despine()
    return fig


# statistics


def plot_bootstrap_distributions(exp):
    """ Plots the bootstrap toi power distributions for each stimulation
    condition and frequency band.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 2 x 3 matplotlib figure. Each subplot contains the bootstrap
        distribution for a particular condition and frequency band.
        Additionally, the estimated toi power and bootstrap 95% CI are
        plotted as vertical lines.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['linewidth']})

    (fig, axs) = plt.subplots(2, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, condition in enumerate(config['conditions']):

        f = '../data/stats/%s_experiment/%s_bootstrap_info.npz' % (exp, condition)
        bootstrap_info = np.load(f)

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
            title = '%s %s Bootstrap Distribution \n Uncorrected p = %.3f'
            ax.set_title(title % (condition, band, p))

    plt.tight_layout()
    sns.despine()
    return fig


def plot_permutation_distributions(exp):
    """ Plots the permutaion toi power difference distributions for each
    pair of stimulation conditions and frequency band.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 2 x 3 matplotlib figure. Each subplot contains the permutation
        distribution for a particular condition comparison and frequency band.
        Additionally, the estimated toi power difference and bootstrap 95% CI
        are plotted as vertical lines.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    comparisons = ["Open-Closed", "Open-Brain", "Brain-Closed"]
    ps = []

    (fig, axs) = plt.subplots(2, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, comp in enumerate(comparisons):
        f = '../data/stats/%s_experiment/%s_%s_permutation_info.npz'
        perm_info = np.load(f % (exp, comp, exp))

        # plot permutation distribution
        ax = axs[0, i]
        sns.distplot(perm_info['alpha_dist'], ax=ax)
        ax.axvline(perm_info['alpha_diff'], color=config['colors'][1])
        title = '%s Alpha Power \n Uncorrected p = %.3f'
        ax.set_title(title % (comp, perm_info['alpha_p_value']))

        ax = axs[1, i]
        sns.distplot(perm_info['beta_dist'], ax=ax)
        ax.axvline(perm_info['beta_diff'], color=config['colors'][1])
        title = '%s Beta Power \n Uncorrected p = %.3f'
        ax.set_title(title % (comp, perm_info['beta_p_value']))

        ps.append(perm_info['alpha_p_value'])
        ps.append(perm_info['beta_p_value'])

    plt.tight_layout()
    sns.despine()

    return fig


def plot_array_permutation_distributions(exp):
    """ Plots the permutaion toi power difference distributions between r
    recording arrays for each stimulation condition and frequency band.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 2 x 3 matplotlib figure. Each subplot contains the permutation
        distribution for a particular condition and frequency band between
        recording arrays. Additionally, the estimated toi power difference and
        permutation 95% CI are plotted as vertical lines.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    (fig, axs) = plt.subplots(2, 3, figsize=(20, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, condition in enumerate(config['conditions']):

        # load condition permutation info
        f = '../data/stats/%s_experiment/' % exp + \
            '%s_array_permutation_info.npz' % condition
        perm_info = np.load(f)

        # iteratively plot band permutation distributions
        for j, band in enumerate(['alpha', 'beta']):
            ax = axs[j, i]
            sns.distplot(perm_info['%s_dist' % band], ax=ax,
                         color=config['colors'][i])
            ax.axvline(perm_info['%s_diff' % band], color=config['colors'][i])
            title = '%s %s Power \n Uncorrected p = %.3f'
            ax.set_title(title % (condition, band.capitalize(),
                                  perm_info['%s_p_value' % band]))

    plt.tight_layout()
    sns.despine()

    return fig


# revision plots

def plot_before_during_after_spectra(exp):
    """ Plots the power spectrum for the 0.5 seconds immediately
    pre-stimulation, the stimulation period, and the 0.5 seconds immediately
    post-stimulation.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 1 x 3 matplotlib figure. Each subplot contains the normalized by sum
        of power spectrum for each condition for a period before, during,
        and after stimulation. Shading is bootstrap standard error.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style='white', font_scale=config['font_scale'],
            rc={"lines.linewidth": config['linewidth']})

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    for i, time_period in enumerate(['Before', 'During', 'After']):

        ax = axs[i]

        for j, condition in enumerate(config['conditions']):

            power, chs, times, freqs = load_power_data(exp, condition)

            power = reduce_array_power(power, chs, config['%s_bad_chs' % exp],
                                       '1', axis=1)

            power = reduce_toi_power(power, times, config[time_period],
                                     axis=-1)

            bootstrap_dist = simple_bootstrap(power, axis=0)

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


def plot_early_vs_late_stim_spectra(exp):
    """ Plots the spectra (averaged TFR power) for the first 5 seconds of the
    stimulation period compared to last 5 seconds of the stimulation period.

    Inputs:
    - exp: main or saline indicating which experiment's data to load and plot

    Outputs:
    - fig: 1 x 3 plot where each plot contains the first and last 5 seconds
    of stimulation spectra for each condition.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    indices = {'Early': (0, 5), 'Late': (5, 10)}
    linestyles = ['-', '--']

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    for i, condition in enumerate(config['conditions']):

        ax = axs[i]
        power, chs, times, freqs = load_power_data(exp, condition)

        # average over trials
        power = power.mean(axis=0)

        # average over array1
        power = reduce_array_power(power, chs, config['%s_bad_chs' % exp],
                                   '1', 0)

        for j, tp in enumerate(['Early', 'Late']):

            # reduce to early or late stim toi
            toi_power = reduce_toi_power(power, times, indices[tp], axis=-1)

            # normalize the spectra
            toi_power /= toi_power.sum()

            # plot the spectra
            ax.plot(freqs, toi_power, color=config['colors'][i],
                    linestyle=linestyles[j])

        # pretty axes
        ax.set_title(condition)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Normalized Power')

    # add legend
    axs[-1].legend(['Early', 'Late'])
    leg = axs[-1].get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')

    plt.tight_layout()
    sns.despine()

    return fig


def plot_controlling_spectra(exp):
    """ Plots the stimulation power spectrum for the bipolar referenced
    electrode that provided the feedback signal for stimulation and a copy
    of the stimulation command stored in the .ns5 files.

    Args:
        exp: The experiment to collect data for. 'main' or 'saline'

    Returns:
        A 1 x 3 matplotlib figure. Each subplot contains the normalized by sum
        of power spectrum for the controlling bipolar referenced electrode
        and a copy of the stimulation command for a particular condition.
        Shading is bootstrap standard error.
    """

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    types = ['ns2', 'ns5']

    # hack the legend to be color agnostic
    legend = ['Neural Recording', 'Stimulation Command']
    axs[2].axvline(-3, color='k', linestyle='--')
    axs[2].axvline(-3, color='k')
    axs[2].legend(legend)

    for i, condition in enumerate(config['conditions']):

        ax = axs[i]

        for typ in types:

            power, chs, times, freqs = load_power_data(exp, condition, typ)

            if typ == 'ns2':
                ch_ix = [ix for ix in np.arange(len(chs))
                         if 'elec1-83' in chs[ix]]
                linestyle = '--'
            else:
                ch_ix = [ix for ix in np.arange(len(chs))
                         if 'ainp2' in chs[ix]]
                linestyle = '-'

            power = power[:, ch_ix, :, :].squeeze()
            power = power.mean(axis=0).mean(axis=-1)
            power = power / power.sum()

            ax.plot(freqs, power, color=config['colors'][i],
                    linestyle=linestyle)
            ax.set_title(condition)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_xlim((freqs[0], freqs[-1]))
            ax.set_ylabel('Normalized Power')

    plt.tight_layout()
    sns.despine()

    return fig
