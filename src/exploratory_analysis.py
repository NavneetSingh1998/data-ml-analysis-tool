def summary_statistics(data):
    """Returns summary statistics of the DataFrame."""
    return data.describe()


def missing_values(data):
    """Returns a DataFrame showing the missing values in the dataset."""
    return data.isnull().sum()


def correlation_matrix(data):
    """Returns the correlation matrix of the DataFrame."""
    return data.corr()


def visualize_distribution(data, column):
    """Visualizes the distribution of a specific column in the DataFrame."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()