"""
This code relate to the paper:
"Privacy Bias in Language Models: A Contextual Integrity-based Auditing Metric"

To appear in the Proceedings of the  Privacy Enhancing Technologies Symposium (PETS), 2026.
"""

import pickle
import lzma
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.tri import Triangulation
from matplotlib.collections import LineCollection
import textwrap


# ---------------------------------------------------------------------
# Custom colormap for heatmaps
# ---------------------------------------------------------------------
# Define a discrete color map for visualization purposes
colors = ["darkred", "lightcoral", "gold", "LimeGreen", "darkgreen"]
c_map = mcolors.ListedColormap(colors)


# ---------------------------------------------------------------------
# Serialization Utilities
# ---------------------------------------------------------------------
def save_object(obj, file_path):
    """
    Save a Python object to disk using pickle.

    Args:
        obj: Any Python object to save.
        file_path (str): File path to save the object.
    """
    try:
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
        print(f"Object saved to {file_path}")
    except Exception as e:
        print(f"Error saving object: {e}")


def load_object(file_path):
    """
    Load a pickle file, supporting .pkl and .pkl.lzma compressed with `lzma -9`.
    """
    try:
        if file_path.endswith(".pkl.lzma"):
            # Pure LZMA (FORMAT_ALONE)
            with lzma.open(file_path, "rb", format=lzma.FORMAT_ALONE) as f:
                obj = pickle.load(f)
        else:
            # Regular pickle file
            with open(file_path, "rb") as f:
                obj = pickle.load(f)

        print(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        print(f"Error loading object: {e}")
        return None


# ---------------------------------------------------------------------
# Heatmap Drawing Function
# ---------------------------------------------------------------------
def heatmap_draw(data, value, ax, fsize=14, pad=0.5):
    data = pd.pivot_table(
        data,
        values=value,
        index=["sender", "transmission"],
        columns=["infotype", "recipient"],
        # aggfunc="mean"
    )
    # display(data)

    # dd
    # Create a mask for missing values

    # fig, ax = plt.subplots()

    # Define discrete colormap
    bounds = [-2, -1, 0, 1, 2, 2.5]
    bound_labels = [
        "Strongly Unacceptable",
        "Somewhat Unacceptable",
        "Neutral",
        "Somewhat Acceptable",
        "Strongly Acceptable",
    ]

    # Create a discrete colormap
    # Create a discrete colormap

    # Create a discrete colormap
    colors = ["darkred", "lightcoral", "yellow", "lightgreen", "darkgreen"]
    c_map = mcolors.ListedColormap(colors)
    # Sample colors from the continuous colormap
    continuous_cmap = plt.get_cmap("coolwarm")  # Choose a continuous colormap
    colors = [
        continuous_cmap(0.5 * (b + (bounds[i + 1] - b) / 2))
        for i, b in enumerate(bounds[:-1])
    ]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(colors))

    # Draw the heatmap
    heatmap = sns.heatmap(
        data,
        cmap=c_map,
        # annot=True,
        norm=norm,
        center=None,  # center is not needed when using discrete colormap
        # mask=mask,
        xticklabels=True,
        yticklabels=True,
        annot_kws={"fontsize": 15},
        ax=ax,
        square=True,
        linewidths=0.1,
        cbar_kws={
            "shrink": 0.65,
            "orientation": "horizontal",
            "pad": pad,
        },  # Ensure colorbar is horizontal
    )

    # Retrieve the colorbar from the heatmap
    colorbar = heatmap.collections[0].colorbar
    # Calculate the midpoints for the ticks
    tick_positions = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]
    # Set colorbar ticks and labels to include the extra value

    colorbar.set_ticks(tick_positions)  # Set ticks to the midpoints
    colorbar.set_ticklabels(bound_labels)
    colorbar.ax.tick_params(labelsize=20)

    # Set tick labels for infotype (bottom)
    # ax.set_xticks(np.arange(len(data.columns.get_level_values(0))) + 0.5, minor=False)
    width = 50
    wrapped_labels = [
        textwrap.fill(label, width) for label in data.columns.get_level_values(0)
    ]
    heatmap.set_xticklabels(wrapped_labels, rotation=45, ha="right", fontsize=fsize)
    # heatmap.set_xticklabels(data.columns.get_level_values(0), rotation=45, ha='right')

    # Set tick labels for recipient (top)
    ax_secondary = heatmap.secondary_xaxis("top")
    ax_secondary.set_xticks(
        np.arange(len(data.columns.get_level_values(1))) + 0.5, minor=False
    )
    ax_secondary.set_xticklabels(
        data.columns.get_level_values(1), fontsize=fsize, rotation=45, ha="left"
    )

    # Set tick labels for transmission (left)
    # ax = heatmap.axes  # Get the primary Axes object for the heatmap
    ax.set_yticks(np.arange(len(data.index.get_level_values(0))) + 0.5)
    ax.set_yticklabels(data.index.get_level_values(0), rotation=0, fontsize=fsize)

    # Set tick labels for transmission (right)
    secondary_ax = ax.secondary_yaxis(
        "right"
    )  # heatmap.figure.axes[1]  # Assuming there is only one Axes object
    cut_labels = [
        label.lower().replace("if the information is", "")
        for label in data.index.get_level_values(1)
    ]

    secondary_ax.set_yticks(np.arange(len(cut_labels)) + 0.5)
    secondary_ax.set_yticklabels(cut_labels, fontsize=fsize)

    # X - vline
    tick_labels = ax.get_xticklabels()
    prev_tick_labels = []
    if len(tick_labels) > 0:
        prev_tick_labels = tick_labels
    else:
        tick_labels = prev_tick_labels
    distance = 0
    prev_label = None
    # print(tick_labels)
    for label in tick_labels:
        if prev_label is None or label.get_text() == prev_label.get_text():
            prev_label = label
            distance += 1
            ax.axvline(x=distance, color="black", linestyle="--", linewidth=1)
        else:
            # Add a vertical line to separate infotype changes
            ax.axvline(x=distance, color="black", linestyle="-", linewidth=2)
            prev_label = label
            distance += 1
    # Y - hline

    # Retrieve x-axis tick positions and labels
    tick_labels = ax.get_yticklabels()
    tick_positions = ax.get_yticks()

    if len(tick_labels) > 0:
        prev_tick_labels = tick_labels
    else:
        tick_labels = prev_tick_labels
    distance = 0
    prev_label = None
    # print(tick_labels)
    for label in tick_labels:
        if prev_label is None or label.get_text() == prev_label.get_text():
            prev_label = label
            ax.axhline(y=distance, color="black", linestyle="--", linewidth=1)
            distance += 1
        else:
            # Add a vertical line to separate infotype changes
            ax.axhline(y=distance, color="black", linestyle="--", linewidth=3)
            prev_label = label
            distance += 1

    # ------ X Labels
    existing_labels = [label.get_text() for label in ax.get_xticklabels()]
    unique_labels, counts = np.unique(existing_labels, return_counts=True)

    tick_positions = (
        np.cumsum(counts) - counts / 2
    )  # Calculate positions in the middle of each group
    new_labels = [" {} ".format(label) for label in unique_labels]

    # Set modified x-axis tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(new_labels, fontsize=fsize)

    ### Y labels

    existing_labels = [
        label.get_text().replace(" (e.g., Siri, Amazon echo)", "")
        for label in ax.get_yticklabels()
    ]
    unique_labels, counts = np.unique(existing_labels, return_counts=True)

    tick_positions = (
        np.cumsum(counts) - counts / 2
    )  # Calculate positions in the middle of each group
    new_labels = [" {} ".format(label) for label in unique_labels]

    # Set modified x-axis tick labels
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(new_labels, fontsize=fsize)

    ax.margins(x=0, y=0)
    ax.set_aspect("equal", "box")  # square cells

    return heatmap


# ---------------------------------------------------------------------
# Triangular Heatmap Grid Function
#  Based on: https://stackoverflow.com/questions/66048529/how-to-create-a-heatmap-where-each-cell-is-divided-into-4-triangles
# ---------------------------------------------------------------------
def triangulation_for_triheatmap(M, N):
    """
    Generate triangulations for a heatmap where each cell is divided into four triangles.

    Args:
        M (int): Number of columns.
        N (int): Number of rows.

    Returns:
        list: List of Triangulation objects for each direction (N, E, S, W).
    """
    # Grid vertices and cell centers
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))

    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)

    # Define triangles in N, E, S, W directions
    trianglesN = [
        (i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
        for j in range(N)
        for i in range(M)
    ]
    trianglesE = [
        (i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
        for j in range(N)
        for i in range(M)
    ]
    trianglesS = [
        (i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
        for j in range(N)
        for i in range(M)
    ]
    trianglesW = [
        (i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
        for j in range(N)
        for i in range(M)
    ]

    return [
        Triangulation(x, y, t) for t in [trianglesN, trianglesE, trianglesS, trianglesW]
    ]


# ---------------------------------------------------------------------
# Tick Label Utilities
# ---------------------------------------------------------------------
def draw_separation_lines(
    ax, tick_labels, orientation="vertical", color="black", linestyle="-", linewidth=2
):
    """
    Draw separation lines between distinct tick labels.

    Args:
        ax (matplotlib.axes.Axes): Axis object.
        tick_labels (list): Tick labels to check for separation.
        orientation (str): 'vertical' or 'horizontal'.
        color (str): Line color.
        linestyle (str): Line style.
        linewidth (float): Line width.
    """
    distance = 0
    prev_label = None

    for label in tick_labels:
        if prev_label is not None and label.get_text() != prev_label.get_text():
            if orientation == "vertical":
                ax.axvline(
                    x=distance - 0.5,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
            else:
                ax.axhline(
                    y=distance - 0.5,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
        else:
            # Dashed line for repeated labels
            if orientation == "vertical":
                ax.axvline(x=distance - 0.5, color=color, linestyle="--", linewidth=1)
            else:
                ax.axhline(y=distance - 0.5, color=color, linestyle="--", linewidth=1)

        prev_label = label
        distance += 1


def set_unique_labels(ax, orientation="x", fontsize=12):
    """
    Remove duplicate tick labels and center labels over grouped ticks.

    Args:
        ax (matplotlib.axes.Axes): Axis object.
        orientation (str): 'x' or 'y'.
        fontsize (int): Font size for tick labels.
    """
    if orientation == "x":
        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    else:
        tick_labels = [label.get_text() for label in ax.get_yticklabels()]

    unique_labels, counts = np.unique(tick_labels, return_counts=True)
    positions = np.cumsum(counts) - counts / 2
    new_labels = [f" {lbl} " for lbl in unique_labels]

    if orientation == "x":
        ax.set_xticks(positions)
        ax.set_xticklabels(new_labels, fontsize=fontsize)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(new_labels, fontsize=fontsize)


# ---------------------------------------------------------------------
# Data Preparation for Heatmaps
# ---------------------------------------------------------------------
def gen_fig_df(data, models, max_response_per_statement, type_of_majority="majority"):
    """
    Generate a pivoted DataFrame suitable for plotting heatmaps.

    Args:
        data (pd.DataFrame): Source DataFrame containing model responses.
        models (list): List of model names to include.
        max_response_per_statement (int): Minimum responses required per statement.
        type_of_majority (str): Column to use as majority ('majority' by default).

    Returns:
        pd.DataFrame: Pivoted DataFrame with statistics (num_responses, variance, mean_score).
    """
    # Filter for selected models and sufficient responses
    fig_df = data.query("model in @models")
    fig_df = fig_df.query(
        "not @pd.isna(@type_of_majority) and num_responses >= @max_response_per_statement"
    )

    # Convert majority column to integer
    fig_df[type_of_majority] = fig_df[type_of_majority].astype(int)

    # Pivot table for heatmap plotting
    fig_df = fig_df.pivot_table(
        index=[
            "sender",
            "infotype",
            "recipient",
            "transmission",
            "Scenario",
            "temperature",
            "dataset",
        ],
        columns=["model"],
        values=type_of_majority,
        aggfunc="first",
    ).reset_index()

    # Identify model columns
    result_columns = fig_df.columns[8:]
    print("Model columns:", result_columns)

    # Calculate statistics per row
    fig_df["num_responses"] = fig_df[result_columns].apply(
        lambda row: sum(~np.isnan(row)), axis=1
    )
    fig_df["variance"] = fig_df[result_columns].apply(lambda row: np.var(row), axis=1)
    fig_df["mean_score"] = fig_df[result_columns].apply(
        lambda row: np.mean(row), axis=1
    )

    return fig_df


# ---------------------------------------------------------------------
# Heatmap Figure Drawing Function
# ---------------------------------------------------------------------
def draw_triangular_heatmap(
    fig_df, models, ci_param_values, dataset="iot", label_font_size=20
):
    """
    Draw a triangular heatmap for the selected models and dataset.
    Supports either two-triangle (duplicated models) or four-triangle configurations.

    Args:
        fig_df (pd.DataFrame): Preprocessed DataFrame with model responses.
        models (list): List of model column names.
        dataset (str): Dataset filter for plotting.
        label_font_size (int): Font size for tick labels.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes: Figure and axes objects.
    """
    # Determine if four-triangle or two-triangle heatmap
    if ci_param_values:
        s, attr, r, tp = ci_param_values

    print(models)
    four_triangles = len(models) == 4
    if not four_triangles:
        # ---- TWO TRINAGLES----
        # Keep all non-model columns
        other_cols = [col for col in fig_df.columns if col not in models]

        # Create copies next to originals and build updated models list
        models_with_copies = []
        for col in models:
            copy_col = f"{col}_copy"
            fig_df[copy_col] = fig_df[col]
            models_with_copies.extend([col, copy_col])

        # Reorder fig_df: other columns first, then models with copies
        fig_df = fig_df[other_cols + models_with_copies]

        # Update models list
        models = models_with_copies

    # Filter DataFrame for the dataset and select relevant columns
    if ci_param_values:
        df_sq = fig_df.query(
            "dataset == @dataset and  sender in @s and infotype in @attr and recipient in @r and transmission not in @tp"
        )[["sender", "infotype", "recipient", "transmission"] + models]
    else:
        df_sq = fig_df.query("dataset == @dataset")[
            ["sender", "infotype", "recipient", "transmission"] + models
        ]

    # Pivot for heatmap plotting
    df_piv = pd.pivot_table(
        df_sq.dropna(),
        index=["sender", "transmission"],
        columns=["infotype", "recipient"],
    )

    M = len(df_piv.columns) // 4
    N = len(df_piv)

    # Extract model values and triangulations
    values = [df_piv[dir] for dir in models]
    triangul = triangulation_for_triheatmap(M, N)
    cmaps = [c_map] * len(models)

    # Define discrete color normalization
    bounds = [-2, -1, 0, 1, 2, 2.5]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(c_map.colors))
    norms = [norm] * len(models)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(80, 40))

    if four_triangles:
        imgs = [
            ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm, ec="white")
            for t, val, cmap, norm in zip(triangul, values, cmaps, norms)
        ]
    else:
        # --- TWO TRIANGLES ---
        imgs = [
            ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm)
            for t, val, cmap, norm in zip(triangul, values, cmaps, norms)
        ]

        # Original square corners (matching the triangulation)
        xv, yv = np.meshgrid(np.arange(0, M), np.arange(0, N))  # bottom-left corners
        xc, yc = np.meshgrid(
            np.arange(0, M), np.arange(0, N)
        )  # centers (already used in triangulation)

        lines = []
        for j in range(N):
            for i in range(M):
                # top-right corner
                x0, y0 = i + 0.5, j + 0.5
                # bottom-left corner
                x1, y1 = i - 0.5, j - 0.5
                lines.append([(x0, y0), (x1, y1)])

        ax.add_collection(LineCollection(lines, colors="white", linewidths=1))

        # ---

    # Set tick labels for recipient (top)
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(np.arange(len(df_piv.columns.get_level_values(2))))
    secax.set_xticklabels(
        df_piv.columns.get_level_values(2),
        fontsize=label_font_size,
        rotation=45,
        ha="left",
    )

    # Apply colors to tick labels
    """
    for tick_label in secax.get_xticklabels(): 
        print(tick_label.get_text())
        if tick_label.get_text() in ['government intelligence agencies','']:
            tick_label.set_color('red')
            tick_label.set_fontweight('bold')
    """

    # Set tick labels for transmission (right)
    secondary_ax = ax.secondary_yaxis("right")
    cut_labels = [
        label.lower().replace("if the information ", "")
        for label in df_piv.index.get_level_values(1)
    ]
    secondary_ax.set_yticks(np.arange(len(df_piv.index.get_level_values(1))))
    secondary_ax.set_yticklabels(cut_labels, fontsize=label_font_size)

    # --- customize depending on the dataset and what needs to be highlighted.
    # Apply colors to tick labels
    """
    for tick_label in secondary_ax.get_yticklabels(): 
        print(tick_label.get_text())
        if tick_label.get_text() in ['is used to serve contextual ads', 'is not stored', 'is used to provide a price discount','is deleted', 'is stored indefinitely', 'is used for advertising']:
            tick_label.set_color('red')
            tick_label.set_fontweight('bold')
            tick_label_position = tick_label.get_position()
    """

    # ===== X labels =====
    ax.set_xticks(range(M))

    # mode=4 triangles
    # xlabels = [label[0] for label in df_piv[models[0]].columns.to_list()]

    xlabels = [
        textwrap.fill(label[0], width=10)
        for label in df_piv[models[0]].columns.to_list()
    ]
    # mode=2 triangles
    # xlabels = [label[0] for label in df_piv[models[0]].columns.to_list()[:M]]

    # no 45 degree
    # ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=label_font_size)
    # no 45 degree
    ax.set_xticklabels(xlabels, ha="center", fontsize=label_font_size)
    # Draw vertical lines to separate different x-axis labels
    draw_separation_lines(
        ax, ax.get_xticklabels(), orientation="vertical", linewidth=4.5
    )

    # Remove redundant labels and set unique x-axis labels
    set_unique_labels(ax, orientation="x", fontsize=label_font_size)

    # ===== Y labels =========
    ax.set_yticks(range(N))
    ylabels = [
        textwrap.fill(label[0].replace(" (e.g., Siri, Amazon echo)", ""), width=20)
        for label in df_piv.index.to_list()
    ]

    ax.set_yticklabels(ylabels, fontsize=label_font_size)

    # Draw horizontal lines to separate different y-axis labels
    draw_separation_lines(
        ax, ax.get_yticklabels(), orientation="horizontal", linewidth=4.5
    )

    # Remove redundant labels and set unique y-axis labels
    set_unique_labels(ax, orientation="y", fontsize=label_font_size)

    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect("equal", "box")  # square cells

    # plt.colorbar(imgs[0], ax=ax)
    # plt.tight_layout()
    plt.show()
