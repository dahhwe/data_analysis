"""Основные функции для построения графиков"""
import pandas as pd
import matplotlib.pyplot as plt
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


def draw_histogram(dataframe: pd.DataFrame, column: str, number: int):
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
