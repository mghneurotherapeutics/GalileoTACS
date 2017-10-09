import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
from data_processing import load_all_data, reduce_toi_power, reduce_array_power
from statistics import simple_bootstrap


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
