from matplotlib import pyplot as plt

class ObesityVisualizer:
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