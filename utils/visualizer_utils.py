from matplotlib import pyplot as plt

from utils.group.group_column_utils import compute_group_proportions


class VisualizerUtils:
    def __init__(self):
        print("[Info] ObesityVisualizer initialized.")

    @staticmethod
    def plot_obesity_distribution(data, title="Obese vs. Non-Obese Individuals"):
        """
        Plot the distribution of obesity classes.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        data['Obese'].value_counts().plot(kind='bar', color=['blue', 'orange'], ax=ax)
        ax.set_xlabel("Obesity Classification", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-Obese", "Obese"], fontsize=10)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_proportions_by_group(proportions, mapping, title, xlabel, ylabel, figsize=(14, 8)):
        """
        Plot proportions for any group (e.g., LocationDesc or Race/Ethnicity).
        """
        fig, ax = plt.subplots(figsize=figsize)
        proportions.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(range(len(proportions.index)))  # Set the tick positions based on the actual data
        ax.set_xticklabels(
            [mapping.get(code, f" {code}") for code in proportions.index],  # Match labels to data
            rotation=90,
            fontsize=10
        )
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    @staticmethod
    def compute_and_visualize_proportions(data, group_col, target, mapping, visualizer, title, xlabel, ylabel,
                                          figsize=None):
        """
        Compute proportions for a specific group column and visualize the results.
        Arguments:
            data: DataFrame containing the data.
            group_col: The column to group by (e.g., 'LocationDesc', 'Race/Ethnicity').
            target: The target column (e.g., 'Obese').
            mapping: Mapping dictionary for the group column.
            visualizer: Visualizer instance with a method to plot proportions.
            title: Title for the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            figsize: Optional figure size for the plot.
        """
        # Compute proportions
        proportions = compute_group_proportions(data, group_col, target)

        # Debug: Ensure group_col and mapping align
        print(f"[Debug] Unique values in '{group_col}':", data[group_col].unique())
        print(f"[Debug] Mapping for {group_col}:", mapping)

        # Visualize proportions
        fig, ax = plt.subplots(figsize=figsize if figsize else (10, 6))
        proportions.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        # Apply mapping to x-axis labels
        ax.set_xticks(range(len(proportions.index)))
        ax.set_xticklabels([mapping.get(code, f" ({code})") for code in proportions.index], rotation=90,
                           fontsize=10)

        plt.tight_layout()
        plt.show()
