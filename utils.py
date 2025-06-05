import pandas as pd
import scipy.stats as ss


def show_value_counts(df: pd.DataFrame, column: str, sort_index: bool = True, ascending: bool = True, decimals: int = 3) -> pd.DataFrame:
    series = df[column]
    value_counts = series.value_counts()

    counts = value_counts.values
    index = value_counts.index
    proportions = counts/len(series)
    percentage = proportions*100

    summary_df = pd.DataFrame({'counts': counts, 'proportions': proportions.round(decimals), 'percentage': percentage.round(3)}, index=index)

    if sort_index:
        return summary_df.sort_index(ascending=ascending)

    return summary_df.sort_values(by="counts", ascending=ascending)


def statistics_central(series: pd.Series) -> None:
    mean = series.mean()
    median = series.median()
    mode = series.mode()

    print("---- Central Tendency Statistics ----")
    print("\tMean:", mean)
    print("\tMedian:", median)
    print("\tMode:", mode.values[0])


def statistics_no_central(series: pd.Series) -> None:
    min = series.min()
    C1 = series.quantile(0.25)
    C2 = series.quantile(0.5)
    C3 = series.quantile(0.75)
    max = series.max()

    print("---- No Central Statistics ----")
    print("\tMin:", min)
    print("\t1st Cuartile:", C1)
    print("\t2nd Cuartile:", C2)
    print("\t3rd Cuartile:", C3)
    print("\tMax:", max)


def statistics_variability(series: pd.Series) -> None:
    range = series.max() - series.min()
    variance = series.var()
    standard_deviation = series.std()
    variance_coeff = ss.variation(series)
    iqr = ss.iqr(series)

    print("---- Variability Statistics ----")
    print("\tRange:", range)
    print("\tVariance:", variance)
    print("\tStandard Deviation:", standard_deviation)
    print(f"\tVariance Coefficient: {round(variance_coeff*100):.2f}%")
    print("\tIQR:", iqr)


def statistics_shape(series: pd.Series) -> None:
    kurtosis_fisher = ss.kurtosis(series, fisher=True)
    kurtosis_pearson = ss.kurtosis(series, fisher=False)
    skewness = ss.skew(series)

    print("---- Shape Statistics ----")
    print("\tSkewness:", skewness)
    print("\tKurtosis (Fisher):", kurtosis_fisher)
    print("\tKurtosis (Pearson):", kurtosis_pearson)


def univariate_statistics(series: pd.Series) -> None:
    statistics_central(series)
    statistics_no_central(series)
    statistics_variability(series)
    statistics_shape(series)


