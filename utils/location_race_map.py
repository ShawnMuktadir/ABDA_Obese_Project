from utils.group.group_column_utils import compute_group_proportions


def generate_mappings(data):
    """
    Generate mappings for LocationDesc and Race/Ethnicity columns.
    Arguments:
        data: DataFrame containing 'LocationDesc' and 'Race/Ethnicity' columns.
    Returns:
        location_mapping: Mapping for LocationDesc.
        race_mapping: Mapping for Race/Ethnicity.
    """
    # Generate mappings
    location_mapping = {code: name for code, name in enumerate(data['LocationDesc'].cat.categories)}
    race_mapping = {code: name for code, name in enumerate(data['Race/Ethnicity'].cat.categories)}

    # Debug: Print the mappings
    print("[Debug] Location Mapping:")
    for code, name in location_mapping.items():
        print(f"{code}: {name}")

    print("[Debug] Race Mapping:")
    for code, name in race_mapping.items():
        print(f"{code}: {name}")

    return location_mapping, race_mapping


def compute_and_visualize_proportions(data, group_col, target, mapping, visualizer, title, xlabel, ylabel, figsize=None):
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

    # Visualize proportions
    visualizer.plot_proportions_by_group(
        proportions,
        mapping,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize
    )

