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

# array comparisons

def plot_array_band_comparison(exp):

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

    for i, c in enumerate(config['conditions']):

        power, chs, times, freqs = load_power_data(exp, c)

        power = baseline_normalize(power, config['baseline'], times)

        # array indices
        arr1_ix = [ix for ix in np.arange(len(chs)) if 'elec1' in chs[ix] and
                   chs[ix] not in config['bad_chs']]
        arr2_ix = [ix for ix in np.arange(len(chs)) if 'elec2' in chs[ix]]

        for j, band in enumerate(['alpha', 'beta']):

            band_power = reduce_band_power(power, freqs, config[band], axis=1)

            for k, arr in enumerate([arr1_ix, arr2_ix]):
                arr_power = band_power[arr, :].mean(axis=0)
                arr_stderr = band_power[arr, :].std(axis=0) / np.sqrt(len(arr))

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
    axs[0, 2].legend(['Array 1', 'Array 2'])
    sns.despine()

    return fig


def plot_array_toi_comparison(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    # plotting initialization
    sns.set(style='white', font_scale=config['font_scale'],
            rc={"lines.linewidth": config['linewidth']})
    fig, axs = plt.subplots(1, 2, figsize=(22, 10))
    plt.subplots_adjust(hspace=.3)

    stat_ys = [-.3, .15, -.4, -.2, .15, -.35]
    stat_hmults = [3, 1.5, 3, 3, 1.5, 3]
    stat_hs = [-.02, .02, -.02, -.02, .02, -.02]

    for i, c in enumerate(config['conditions']):

        f = '../stats/%s_experiment/%s_array_permutation_info.npz'
        perm_info = np.load(f % (exp, c))

        power, chs, times, freqs = load_power_data(exp, c)

        power = baseline_normalize(power, config['baseline'], times)

        # array indices
        arr1_ix = [ix for ix in np.arange(len(chs)) if 'elec1' in chs[ix] and
                   chs[ix] not in config['bad_chs']]
        arr2_ix = [ix for ix in np.arange(len(chs)) if 'elec2' in chs[ix]]

        for j, band in enumerate(['alpha', 'beta']):

            band_power = reduce_band_power(power, freqs, config[band], axis=1)
            toi_power = reduce_toi_power(band_power, times, config['toi'],
                                         axis=-1)

            for k, arr in enumerate([arr1_ix, arr2_ix]):
                arr_power = toi_power[arr].mean(axis=0)
                arr_stderr = toi_power[arr].std(axis=0) / np.sqrt(len(arr))

                bar_tick = i * 2 + k * .8

                if k == 0:
                    axs[j].bar(bar_tick, arr_power, color=config['colors'][i],
                               yerr=arr_stderr, ecolor='k')
                else:
                    axs[j].bar(bar_tick, arr_power, facecolor='none',
                               edgecolor=config['colors'][i], linewidth=4,
                               yerr=arr_stderr, ecolor='k', hatch='/')

            # pretty axis
            axs[j].set_title('%s Power' % band.capitalize(), y=1.05)
            axs[j].set_xticks([x + .8 for x in [0, 2, 4]])
            axs[j].set_xticklabels(config['conditions'])
            axs[j].set_ylim((-.5, .2))
            axs[j].set_xlim((0, 6))
            axs[j].set_ylabel('dB Change From Baseline')
            axs[j].axhline(0, color='k', label='_nolegend_')

            # statistical annotation
            p = perm_info['%s_p_value' % band]
            if p < .001:
                p = 'p < .001'
            else:
                p = 'p = %.03f' % p
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

        f = './stats/saline_experiment/%s_bootstrap_info.npz' % condition
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

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    comparisons = ["Open-Closed", "Open-Brain", "Brain-Closed"]
    ps = []

    (fig, axs) = plt.subplots(2, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, comp in enumerate(comparisons):
        f = './stats/%s_experiment/%s_%s_permutation_info.npz'
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


# statistics

def plot_array_permutation_distributions(exp):

    with open('./experiment_config.json', 'r') as f:
        config = json.load(f)

    sns.set(style="white", font_scale=config['font_scale'],
            rc={"lines.linewidth": config['font_scale']})

    (fig, axs) = plt.subplots(2, 3, figsize=(20, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, condition in enumerate(config['conditions']):

        # load condition permutation info
        f = '../stats/%s_experiment/' % exp + \
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


# revisions

def plot_early_vs_late_stim_spectra(exp):
    """ Plots the spectra (averaged TFR power) for the first 5 seconds of the
    stimulation period compared to last 5 seconds of the stimulation period.

    Inputs:
    - exp: main or saline indicating which experiment's data to load and plot

    Outputs:
    - fig: 1 x 3 plot where each plot contains the first and last 5 seconds
    of stimulation spectra for each condition
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
        power = reduce_array_power(power, chs, config['bad_chs'], '1', 0)

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


