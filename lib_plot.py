import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List


def concat_columns_for_color(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df["color"] = df[columns].agg(lambda xs: "_".join(map(str, xs)), axis=1)
    return df


def prepare_for_plotting(
    df: pd.DataFrame, column: str, groupby: List[str] = ["color", "lambda"]
) -> pd.DataFrame:
    df = wnc(df, column)
    df = df[[column, *groupby]]
    g = df.groupby(groupby)

    mean = g.quantile(0.5, interpolation="linear")
    top = (g.quantile(0.9, interpolation="nearest") - mean).rename(
        columns={column: "top"}
    )
    bot = (mean - g.quantile(0.1, interpolation="nearest")).rename(
        columns={column: "bottom"}
    )
    ret = mean.join(top).join(bot)
    ret = ret.reset_index()
    return ret


def plot_with_color(fig: go.Figure, df: pd.DataFrame, column: str) -> None:
    for color in df["color"].unique():
        fdf = df[df["color"] == color]
        fig.add_trace(
            go.Scatter(
                x=fdf["lambda"] + 1,
                y=fdf[column],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=fdf["top"],
                    arrayminus=fdf["bottom"],
                ),
                name=color,
            )
        )


def end_to_end_plot(
    fig: go.Figure,
    df: pd.DataFrame,
    data_column: str,
    group_by_columns: List[str],
    title_text: str,
) -> go.Figure:
    df = concat_columns_for_color(wnc(df, data_column), group_by_columns)
    df = prepare_for_plotting(df, data_column)
    plot_with_color(fig, df, data_column)
    fig.update_layout(xaxis_type="log", title_text=title_text, xaxis_title="lambda + 1")
    return fig


def row_multimap(df: pd.DataFrame, f) -> pd.DataFrame:
    def f_map(row):
        ret = f(row)
        ret = [r.to_frame().T for r in ret]
        ret = pd.concat(ret)
        return ret

    rows = []
    for r in df.apply(f_map, axis=1):
        rows.append(r)
    df_new = pd.concat(rows).reset_index(drop=True)
    return df_ne


def expand_network(df: pd.DataFrame, order_by_column: str) -> pd.DataFrame:
    def f_map(row):
        n_networks = 2
        if row[order_by_column][0] <= row[order_by_column][1]:
            net_0 = 0
            net_1 = 1
        else:
            net_0 = 1
            net_1 = 0

        ret = [row.copy() for _ in range(n_networks)]

        ret[0]["network"] = "net0"
        ret[1]["network"] = "net1"

        for k, v in row.items():
            if type(v) == type([]) or type(v) == type((0, 0)):
                ret[0][k] = v[net_0]
                ret[1][k] = v[net_1]
        # assert ret[0][order_by_column] <= ret[1][order_by_column]
        return ret

    return row_multimap(df, f_map)


def split_train_test(df: pd.DataFrame, column: str) -> pd.DataFrame:
    def f_map(row):
        train_row = row.copy()
        train_row["distribution"] = "train"
        test_row = row.copy()
        test_row["distribution"] = "test"

        train_row[column] = row[f"train_{column}"]
        test_row[column] = row[f"test_{column}"]

        return [train_row, test_row]
        ret = [r.to_frame().T for r in ret]
        ret = pd.concat(ret)
        return ret

    df = row_multimap(df, f_map)
    df = df.drop([f"train_{column}", f"test_{column}"], axis=1)
    df[column] = df[column].apply(pd.to_numeric)
    return df


def wnc(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].apply(pd.to_numeric)
    return df
