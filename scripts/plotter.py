#!/bin/python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

def plot_custom_oncoplot(matrix_df: pd.DataFrame, title: str = "OncoPrint"):
    """
    matrix_df column schema: ['sample', 'gene', 'alteration']
    alteration values: Missense_Mutation, Nonsense_Mutation, Frame_Shift, Splice_Site, Fusion, Multi_Hit, In_Frame
    """
    if matrix_df.empty:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No alterations found", ha='center', va='center', fontsize=20)
        plt.axis('off')
        return

    # Define alteration types and colors
    alteration_colors = {
        'Missense_Mutation': '#008000',    
        'Nonsense_Mutation': '#000000',   
        'Frame_Shift':       '#993300',   
        'Splice_Site':       '#6600CC',   
        'In_Frame_Del':      '#00CCFF',  
        'In_Frame_Ins':      '#00CCFF',
        'Fusion':            '#FF00FF',   
        'Multi_Hit':         '#808080',    
    }

    df = matrix_df.copy()
    df['count'] = 1

    # Sort genes by alteration frequency
    gene_order = (df.groupby('gene')['sample'].nunique() / df['sample'].nunique() * 100)
    gene_order = gene_order.sort_values(ascending=False).index.tolist()

    # Sort samples by total alterations
    sample_order = df['sample'].value_counts().index.tolist()

    df = df.set_index(['gene', 'sample']).sort_index()
    df = df.loc[gene_order, :]  # ensure gene order

    # Build grid
    genes = gene_order
    samples = sample_order

    fig = plt.figure(figsize=(max(12, len(samples) * 0.15), max(6, len(genes) * 0.35)))
    gs = fig.add_gridspec(10, 1, height_ratios=[0.5, 0.2, 6, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Main plot
    ax_main = fig.add_subplot(gs[2, 0])

    y_positions = np.arange(len(genes))
    x_positions = np.arange(len(samples))

    plotted = set()

    for y, gene in enumerate(genes):
        for x, sample in enumerate(samples):
            alts = df.loc[(df.index.get_level_values(0) == gene) &
                          (df.index.get_level_values(1) == sample), 'alteration']
            if alts.empty:
                continue

            # Priority: Fusion > Multi > Frame > etc.
            priority = ['Fusion', 'Multi_Hit', 'Frame_Shift', 'Nonsense_Mutation', 'Splice_Site', 'Missense_Mutation']
            alt = alts.iloc[0]
            for p in priority:
                if p in alts.values:
                    alt = p
                    break

            color = alteration_colors.get(alt, '#A0A0A0')
            if alt == 'Multi_Hit':
                # Draw two overlapping rectangles
                rect1 = mpatches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, facecolor='#008000', edgecolor='black', linewidth=0.5)
                rect2 = mpatches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, facecolor='#993300', alpha=0.6)
                ax_main.add_patch(rect1)
                ax_main.add_patch(rect2)
            else:
                rect = mpatches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                          facecolor=color, edgecolor='black', linewidth=0.3)
                ax_main.add_patch(rect)

            plotted.add((gene, sample))

    ax_main.set_yticks(y_positions)
    ax_main.set_yticklabels(genes)
    ax_main.set_xticks(x_positions[::max(1, len(x_positions)//30)])  # Avoid crowding
    ax_main.set_xticklabels(samples[::max(1, len(x_positions)//30)], rotation=90, ha='center')
    ax_main.set_xlim(-0.5, len(samples) - 0.5)
    ax_main.set_ylim(-0.5, len(genes) - 0.5)
    ax_main.invert_yaxis()
    ax_main.grid(False)
    ax_main.set_title(title, fontsize=14, pad=30)

    # Alteration frequency barplot (left)
    ax_bar = fig.add_subplot(gs[2, 0], sharey=ax_main, frameon=False)
    ax_bar.barh(y_positions, [gene_order.tolist().index(g) + 1 for g in genes],
                height=0.8, color='#2c7bb6')
    ax_bar.set_xticks([])
    ax_bar.set_xlim(0, len(samples))
    ax_bar.invert_xaxis()
    for y, gene in enumerate(genes):
        count = df.loc[gene].shape[0] if gene in df.index else 0
        pct = count / len(samples) * 100
        ax_bar.text(2, y, f"{pct:.1f}%", va='center', ha='left', fontsize=9, color='black')

    # Legend
    legend_elements = [
        mpatches.Patch(color=v, label=k.replace('_', ' ')) for k, v in alteration_colors.items()
        if k != 'Multi_Hit'
    ]
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='black', label='Multi-Hit (overlaid)'))
    ax_main.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', title="Alteration Type")

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()


# CIRCOS PLOT
# TODO: Most prob. using https://github.com/moshi4/pyCirclize

# MULTIVARIATE PLOTS
# TODO: Infer variables to plot from input query
# Also, based on filtering critera, give dynamic selection of possible plots.
