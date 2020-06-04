import numpy as np

# Label for the output
STATS = 'batch size,median,mean,std_dev,min_time,'\
        'max_time,quantile_10,quantile_90'.split(',')

def get_header(padding: int = 15):
    """Prepares a pretty header for output"""
    header = [f'{x:>{padding-1}} ' for x in STATS]
    return '|'.join(header)

def get_underline(padding: int = 15):
    """Line below header"""
    underline = '+'.join(['-' * (padding)] * len(STATS))
    return underline

def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    time_array = np.array(time_list)

    median = np.median(time_array)
    mean = np.mean(time_array)
    std_dev = np.std(time_array)
    max_time = np.amax(time_array)
    min_time = np.amin(time_array)
    quantile_10 = np.quantile(time_array, 0.1)
    quantile_90 = np.quantile(time_array, 0.9)

    return (median, mean, std_dev, min_time, 
        max_time, quantile_10, quantile_90)


def format_stats(batch_size, stats, padding: int = 15):
    """Makes stats pretty for printing"""
    stats_str = [f'{x:.2f}' for x in stats]
    stats_padding = [f'{x:>{padding-1}} ' for x in stats_str]
    row = [f'{batch_size:>{padding-1}} '] + stats_padding
    return '|'.join(row)
