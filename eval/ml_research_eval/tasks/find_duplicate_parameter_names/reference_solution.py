def find_duplicate_parameter_names(named_parameters):
    seen = {}
    duplicates = []
    for name, param in named_parameters:
        identifier = id(param)
        if identifier in seen:
            duplicates.append(name)
        else:
            seen[identifier] = name
    return duplicates
