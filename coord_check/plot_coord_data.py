from copy import copy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_coord_data(df, y='l1', save_to=None, suptitle=None, x='width', hue='module',
                    legend='full', name_contains=None, name_not_contains=None, module_list=None,
                    loglog=True, logbase=2, face_color=None, subplot_width=5,
                    subplot_height=4):
    '''Plot coord check data `df` obtained from `get_coord_data`.

    Input:
        df:
            a pandas DataFrame obtained from `get_coord_data`
        y:
            the column of `df` to plot on the y-axis. Default: `'l1'`
        save_to:
            path to save the resulting figure, or None. Default: None.
        suptitle:
            The title of the entire figure.
        x:
            the column of `df` to plot on the x-axis. Default: `'width'`
        hue:
            the column of `df` to represent as color. Default: `'module'`
        legend:
            'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.
        name_contains, name_not_contains:
            only plot modules whose name contains `name_contains` and does not contain `name_not_contains`
        module_list:
            only plot modules that are given in the list, overrides `name_contains` and `name_not_contains`
        loglog:
            whether to use loglog scale. Default: True
        logbase:
            the log base, if using loglog scale. Default: 2
        face_color:
            background color of the plot. Default: None (which means white)
        subplot_width, subplot_height:
            The width and height for each timestep's subplot. More precisely,
            the figure size will be
                `(subplot_width*number_of_time_steps, subplot_height)`.
            Default: 5, 4

    Output:
        the `matplotlib` figure object
    '''
    ### preprocessing
    df = copy(df)
    # nn.Sequential has name '', which duplicates the output layer
    df = df[df.module != '']
    if module_list is not None:
        df = df[df['module'].isin(module_list)]
    else:
        if name_contains is not None:
            df = df[df['module'].str.contains(name_contains)]
        if name_not_contains is not None:
            df = df[~(df['module'].str.contains(name_not_contains))]
    # for nn.Sequential, module names are numerical
    try:
        df['module'] = pd.to_numeric(df['module'])
    except ValueError:
        pass

    ts = df.t.unique()

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    plt.rcParams.update({
        'font.size': 15
    })

    def tight_layout(plt):
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ### plot
    fig = plt.figure(figsize=(subplot_width * len(ts), subplot_height))
    hue_order = sorted(set(df['module']))
    if face_color is not None:
        fig.patch.set_facecolor(face_color)
      
    ymin, ymax = 0.5*min(df[y]), 2*max(df[y])
    ymin, ymax = 0.5*min(df[y]), 1.1*max(df[y])
                      
    for t in ts:
        t = int(t)
        plt.subplot(1, len(ts), t)
        sns.lineplot(x=x, y=y, data=df[df.t == t], hue=hue, hue_order=hue_order, legend=legend if t == 1 else None)
        plt.xlabel('width', fontsize=20)
        plt.title(f't={t}', fontsize=20)
        if t != 1:
            plt.ylabel('')
        if loglog:
            plt.loglog(base=logbase)
        ax = plt.gca()
        ax.set_ylim([ymin, ymax])
    if suptitle:
        plt.suptitle(suptitle)
    tight_layout(plt)
    if save_to is not None:
        plt.savefig(save_to)
        print(f'coord check plot saved to {save_to}')

    return fig
