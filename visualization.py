import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df, save_path=None):
    """Plot a heatmap of feature correlations."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_pairplot(df, features, title, save_path=None):
    """Plot a pairplot for selected features."""
    sns.pairplot(data=df[features], diag_kind=None)
    plt.suptitle(title, size=16)
    if save_path:
        plt.savefig(save_path)
    plt.show()