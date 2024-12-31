
def generate_mappings(data):
    """
    Generate mappings for LocationDesc, Race/Ethnicity, and Education columns.
    Arguments:
        data: DataFrame containing 'LocationDesc', 'Race/Ethnicity', and 'Education' columns.
    Returns:
        location_mapping: Mapping for LocationDesc.
        race_mapping: Mapping for Race/Ethnicity.
        education_mapping: Mapping for Education.
    """
    # Generate mappings
    location_mapping = {code: name for code, name in enumerate(data['LocationDesc'].cat.categories)}
    race_mapping = {code: name for code, name in enumerate(data['Race/Ethnicity'].cat.categories)}
    education_mapping = {code: name for code, name in enumerate(data['Education'].cat.categories)}

    # Debug: Print the mappings
    print("[Debug] Location Mapping:")
    for code, name in location_mapping.items():
        print(f"{code}: {name}")

    print("[Debug] Race Mapping:")
    for code, name in race_mapping.items():
        print(f"{code}: {name}")

    if education_mapping:
        print("[Debug] Education Mapping:")
        for code, name in education_mapping.items():
            print(f"{code}: {name}")

    return location_mapping, race_mapping, education_mapping






