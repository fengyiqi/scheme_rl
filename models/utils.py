import csv


def write_to_csv(file_name: str, data: list):
    """
    a helper function to write data into csv file
    :param file_name: csv file name
    :param data: a n-d array that needs to write into csv file
    """
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
