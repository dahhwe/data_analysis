"""Основные функции для построения графиков"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sigmaclip
import seaborn as sns


def draw_charts(dataframe: pd.DataFrame, column: str, var: str, fig_size=(15, 5)) -> None:
    """
    Построение столбчатой и круговой диаграммы.
    :param dataframe:
    :param column:
    :param var:
    :param fig_size: tuple, optional, default (15, 5)
        Size of the figure (width, height) in inches.
    :return:
    """
    count = dataframe[column].value_counts()
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    axs[0].bar(x=count.index, height=count.values)
    axs[0].set(title=f"Столбчатая диаграмма распределения {var} в выборке",
               xlabel='значения выборки', ylabel='частота')

    axs[1].pie(count.values, labels=count.index, autopct='%1.1f%%')
    axs[1].legend(bbox_to_anchor=(0.9, 1))
    axs[1].set(title=f"Круговая диаграмма распределения {var} в выборке")

    plt.show()


def quartile_method(quartile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Метод квартилей
    :param quartile_df: датафрейм для применения метода квартилей
    :return: измененный датафрейм, с отсеченными значениями
    """
    num_cols = quartile_df.select_dtypes(include=['float']).columns
    q1 = quartile_df[num_cols].quantile(0.25)
    q3 = quartile_df[num_cols].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_df = quartile_df[
        ~((quartile_df[num_cols] < lower_bound) | (quartile_df[num_cols] > upper_bound)).any(axis=1)]
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def sigma_method(sigma_df: pd.DataFrame):
    """
    Метод сигм
    :param sigma_df: датафрейм для применения метода сигм
    :return: измененный датафрейм, с отсеченными значениями
    """
    numerical_column = sigma_df.select_dtypes(include=['float']).columns
    for column in numerical_column:
        data = sigma_df[column].dropna()
        clean_data, low, high = sigmaclip(data, low=3, high=3)
        sigma_df = sigma_df.loc[(sigma_df[column].isin(clean_data)) | (sigma_df[column].isna())]

    sigma_df = sigma_df.reset_index()
    sigma_df.pop('index')
    return sigma_df


def draw_histogram(dataframe: pd.DataFrame, column: str, number: int) -> None:
    """
    Построение гистограммы, оценки плотности распределения и диаграммы
    "ящик с усами".
    :param dataframe:
    :param column:
    :param number:
    :return:
    """
    num1 = dataframe[column].dropna()
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    axs[0].hist(num1, bins=100, density=True, alpha=0.5)
    axs[0].set(title=f"Гистограмма числового параметра #{number}",
               xlabel='значения выборки', ylabel='частота')

    axs[1].boxplot(x=num1)
    axs[1].set(title=f"Диаграмма 'ящик с усами' числового параметра #{number}",
               xlabel='номер выборки', ylabel='разброс значений')

    axs[2].set(title=f"Оценка функции плотности числового параметра #{number}",
               xlabel='значения выборки', ylabel='вероятность')
    dataframe[column].plot.kde(ax=axs[2], bw_method=0.1)

    plt.show()


def remove_outliers(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    _, low, upp = sigmaclip(dataframe[column], 3, 3)
    dropped_values = dataframe[column][(dataframe[column] < low) | (dataframe[column] > upp)]
    dataframe = dataframe.drop(dropped_values.index)
    return dataframe


def visualisation(df, data_type, data_cols, data_titles=None):
    def visual_func(data, title):
        fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))
        fig.suptitle(title, fontsize=20)

        if data_type == 'qualitative':
            counts = data.value_counts()
            bar_axe, pie_axe = axes

            bar_axe.bar(counts.index, counts.values)
            bar_axe.set_xticks(range(len(counts.index)))
            bar_axe.set_xticklabels(counts.index)
            bar_axe.set_xlabel('Варианты')
            bar_axe.set_ylabel('Количество')
            bar_axe.set_title('Столбчатая диаграмма')

            pie_axe.pie(counts.values, labels=counts.index, autopct='%1.2f%%')
            pie_axe.set_title('Круговая диаграмма')
        else:
            sns.histplot(data, kde=True, ax=axes[0])
            axes[0].set_xlabel('Значения')
            axes[0].set_ylabel('Относительная плотность')
            axes[0].set_title('Гистограмма с графиком функции распределения')

            sns.boxplot(data, ax=axes[1], orient='h')
            axes[1].set_xlabel('Значения')
            axes[1].set_ylabel('')
            axes[1].set_title('Ящик с усами')

        plt.show()

    if data_titles is None:
        data_titles = data_cols

    for col, title in zip(data_cols, data_titles):
        visual_func(df[col], title)


def draw_quantitative(
    series: pd.Series,
    title: str,
    x_label: str,
    kde_bandwidth: float = 0.5,
    boxplot_orientation: str = 'h'
) -> None:
    """
    Function to draw quantitative plots
    :param series: Series to draw
    :param title: Plot title
    :param x_label: Label for the x-axis
    :param kde_bandwidth: Bandwidth for the kernel density estimation
    :param boxplot_orientation: Orientation of the boxplot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 5), dpi=100)
    fig.suptitle(title, fontsize=20)
    ax1.set_ylabel('Relative frequency density')
    ax1.set_xlabel(x_label)
    ax2.set_ylabel('Relative frequencies')
    ax2.set_xlabel(x_label)
    ax3.set_xlabel('Values')
    ax1.hist(series)
    sns.kdeplot(data=series, bw_method=kde_bandwidth, ax=ax2)
    sns.boxplot(data=series, ax=ax3, orient=boxplot_orientation)
    plt.show()