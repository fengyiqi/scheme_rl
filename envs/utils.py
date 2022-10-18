def normalize(value, bounds):
    normalized = (value - bounds[0]) / (bounds[1] - bounds[0])
    return normalized
