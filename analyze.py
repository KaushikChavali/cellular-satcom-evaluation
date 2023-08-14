#!/usr/bin/env python3

# cellular-satcom-evaluation : Evaluation sciprts for the Multipath
# Cellular and SATCOM Emulation Testbed.
#
# Copyright (C) 2023 Kaushik Chavali
# 
# This file is part of the cellular-satcom-evaluation.
#
# cellular-satcom-evaluation is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License 
# as published by the Free Software Foundation, either version 3 of 
# the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import multiprocessing as mp
import os
import sched
from time import time
from typing import Optional, Dict, Tuple, Iterable, List, Callable, Generator

import numpy as np
import pandas as pd
from pygnuplot import gnuplot

from common import MeasureType, GRAPH_DIR, DATA_DIR, logger

LINE_COLORS = ['000000', 'FF0000', '9400D3', '0000FF', '006400', 'FF8C00', 'FFD700', '00FFFF', '00FF7F',
               'FFA500', 'ADFF2F', 'EE82EE', '4169E1', 'FF1493', 'FFC0CB', '2E8B57']
POINT_TYPES = [2, 4, 8, 10, 6, 12, 9, 11, 13, 15, 17, 20, 22, 33, 34, 50]
SI = {
    'K': 2 ** 10, 'M': 2 ** 20, 'G': 2 ** 30, 'T': 2 ** 40, 'P': 2 ** 50, 'E': 2 ** 60,
    'k': 10 ** 3, 'm': 10 ** 6, 'g': 10 ** 9, 't': 10 ** 12, 'p': 10 ** 15, 'e': 10 ** 18,
}

GRAPH_PLOT_SIZE_CM = (12, 6)
GRAPH_PLOT_SECONDS = 1800
GRAPH_PLOT_RTT_SECONDS = 100
GRAPH_X_BUCKET = 0.1
VALUE_PLOT_SIZE_CM = (8, 8)
MATRIX_KEY_SIZE = 0.12
GRAPH_Y_RANGE_RTT = 1000
GRAPH_Y_RANGE_GOODPUT = 20
GRAPH_Y_RANGE_LOSS = 1
GRAPH_Y_RANGE_RETRANSMITS = 1
GRAPH_Y_RANGE_DUP_ACKS = 100
GRAPH_Y_RANGE_FAST_RETRANSMITS = 100
GRAPH_Y_RANGE_OFO_SEGEMETNS = 100
# For sideways figure
MATRIX_SUBPLOT_SIZE_CM = (GRAPH_PLOT_SIZE_CM[0] * 0.75, GRAPH_PLOT_SIZE_CM[1] * 0.75)
# For normal figure
# MATRIX_SUBPLOT_SIZE_CM = (GRAPH_PLOT_SIZE_CM[0] * 0.5, GRAPH_PLOT_SIZE_CM[1] * 0.75)

DEBUG_GNUPLOT = False

PointMap = Dict[any, int]
LineMap = Dict[any, str]
FileTuple = Tuple[any, ...]
DataTuple = Tuple[any, ...]


def create_output_dirs(out_dir: str):
    """
    Creates output directories (GRAPH_DIR and DATA_DIR) of the analysis if the don't already exist.

    :param out_dir: The output folder set via the command line
    :return:
    """
    graph_dir = os.path.join(out_dir, GRAPH_DIR)
    data_dir = os.path.join(out_dir, DATA_DIR)

    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def unique_cartesian_product(df: pd.DataFrame, *col_names: str) -> Generator[Tuple[any, ...], None, None]:
    """
    Generates the cartesian product of the unique values for each column in the dataframe.

    :param df: The dataframe to read the unique values per column from.
    :param col_names: The names of the columns to use for the cartesian product.
    :return: A generator for value tuples based on the specified columns in the given dataframe.
    """

    if len(col_names) < 1:
        yield tuple()
        return

    unique_vals = tuple(list(df[name].unique()) for name in col_names)
    vids = [0 for _ in col_names]

    while vids[0] < len(unique_vals[0]):
        yield tuple(unique_vals[cid][vid] for cid, vid in enumerate(vids))
        # Increment
        for cid in range(len(col_names) - 1, -1, -1):
            vids[cid] += 1
            if vids[cid] < len(unique_vals[cid]):
                break
            elif cid != 0:
                vids[cid] = 0


def scheduler_key(mp_sched: str):
    """
    Provides the key for sorting schedulers in alphabetical order.

    :param mp_sched: The MPTCP scheduler name to sort
    :return:
    """
    try:
        return ['NONE', 'BLEST', 'DEFAULT', 'REDUNDANT', 'ROUNDROBIN'].index(mp_sched.upper())
    except ValueError:
        return -1


def mp_cc_key(mp_cc: str):
    """
    Provides the key for sorting MPTCP CC.

    :param mp_cc: The MPTCP CC name to sort
    :return:
    """
    try:
        return ['NONE', 'RENO', 'CUBIC', 'LIA', 'OLIA', 'BALIA', 'WVEGAS'].index(mp_cc.upper())
    except ValueError:
        return -1


def sprint_tuple(col_names: List[str], col_values: Tuple[any, ...]) -> str:
    """
    Format a tuple into a printable string.

    :param col_names: Names of the columns in the tuple
    :param col_values: Tuple values
    :return: A string with all column names and tuple values
    """
    return ', '.join(["%s=%s" % (col, str(val)) for col, val in zip(col_names, col_values)])


def sprint_scheduler_name(mp_sched: str) -> str:
    """
    Format the MPTCP scheduler name to the shortest possible values.

    :param mp_sched: Scheduler string to format
    :return: Formatted scheduler name
    """

    if mp_sched == "default":
        return "def"
    elif mp_sched == "roundrobin":
        return "rr"
    elif mp_sched == "redundant":
        return "red"
    else:
        return mp_sched


def apply_si(val: str) -> int:
    """
    Parse an integer from a string with an optional SI suffix.

    :param val: The string to parse the integer from.
    :return: An integer with the si suffix applied.
    """

    if val.isdecimal():
        return int(val)
    if val[:-1].isdecimal():
        return int(val[:-1]) * SI.get(val[-1:], -1)


def filter_graph_data(df: pd.DataFrame, x_col: str, x_range: Optional[Tuple[int, int]], file_cols: List[str],
                      file_tuple: FileTuple) -> Optional[pd.DataFrame]:
    """
    Filter data relevant for the graph from the dataframe.

    :param df: The dataframe to filter
    :param x_col: Name of the column that has the data for the x-axis, only used if x_range is given
    :param x_range: (min, max) tuple for filtering the values for the x-axis, or None for no filter
    :param file_cols: Column names that define values for which separate graphs are generated
    :param file_tuple: The set of values for the file_cols that are used in this graph
    :return:
    """

    gdf_filter = True

    if x_range is not None:
        gdf_filter = (df[x_col] >= x_range[0]) & (df[x_col] < x_range[1])

    for col_name, col_val in zip(file_cols, file_tuple):
        gdf_filter &= df[col_name] == col_val

    gdf = df.loc[gdf_filter]
    return None if gdf.empty else gdf


def get_point_type(point_map: PointMap, val: any):
    """
    Selects the gnuplot 'pointtype' based on the given value. The map ensures, that the same values give the same types.

    :param point_map: The map to lookup point types from
    :param val: The value to lookup or generate a point type for
    :return:
    """

    if val not in point_map:
        idx = len(point_map)
        # Use default value if more point types than specified are requested
        point_map[val] = 7 if idx >= len(POINT_TYPES) else POINT_TYPES[idx]

    return point_map[val]


def get_line_color(line_map: LineMap, val: any):
    """
    Selects the gnuplot 'linecolor' based on the given value. The map ensures, that the same values give the same color.

    :param line_map: The map to lookup line colors from
    :param val: The value to lookup or generate a line color for
    :return:
    """

    if val not in line_map:
        idx = len(line_map)
        # Use default value if more line colors than specified are requested
        line_map[val] = '7F7F7F' if idx >= len(LINE_COLORS) else LINE_COLORS[idx]

    return line_map[val]


def prepare_time_series_graph_data(df: pd.DataFrame, x_col: str, y_col: str, x_range: Optional[Tuple[int, int]],
                                   x_bucket: Optional[float], y_div: float, extra_title_col: str, file_cols: List[str],
                                   file_tuple: FileTuple, data_cols: List[str], point_map: PointMap, line_map: LineMap,
                                   point_type_indices: List[int], line_color_indices: List[int],
                                   format_data_title: Callable[[DataTuple], str],
                                   data_tuple_key_transform: Callable[[DataTuple], any] = lambda x: x,
                                   plot_confidence_area: bool = False
                                   ) -> Optional[Tuple[pd.DataFrame, List[str], List[tuple]]]:
    """
    Prepare data to be used in a time series graph.

    :param df: The dataframe to read the data from
    :param x_col: Name of the column that has the data for the x-axis
    :param y_col: Name of the column that has the data for the y-axis
    :param x_range: (min, max) tuple for filtering the values for the x-axis
    :param x_bucket: Size of the bucket to use for aggregating data on the x-axis
    :param y_div: Number to divide the values on the y-axis by before plotting
    :param extra_title_col: Name of the column that holds a string prefix for the data title
    :param file_cols: Column names that define values for which separate graphs are generated
    :param file_tuple: The set of values for the file_cols that are used in this graph
    :param data_cols: Column names of the columns used for the data lines
    :param point_map: Map that ensures identical point types for same data lines
    :param line_map: Map that ensures identical line colors for same data lines
    :param point_type_indices: Indices of file_cols used to determine point type
    :param line_color_indices: Indices of file_cols used to determine line color
    :param format_data_title: Function to format the title of a data line, receives a data_tuple
    :param data_tuple_key_transform: Function to transform a data tuple as key for the sort method
    :param plot_confidence_area: Whether to add the confidence area to the plot commands
    :return: A tuple consisting of a dataframe that holds all data for the graph, a list of plot commands and a list of
    data_tuples that will be plotted in the graph. If at some point there are no data left and therefore plotting the
    graph would be useless, None is returned.
    """

    # Filter data for graph
    gdf = filter_graph_data(df, x_col, x_range, file_cols, file_tuple)
    if gdf is None or gdf.empty:
        return None
    gdf = pd.DataFrame(gdf)

    if x_bucket is not None:
        if x_range is not None:
            start, end = x_range
        else:
            start = gdf[x_col].min()
            end = gdf[x_col].max()
        # Start one bucket earlier to add zero data point (lines start at origin)
        # End one bucket after since each bucket is defined as [a;b) with a being the name of the bucket
        buckets = np.arange(start=start - x_bucket, stop=end + x_bucket, step=x_bucket)
        gdf[x_col] = pd.cut(gdf[x_col], buckets, labels=buckets[1:])

    # Calculate mean average per y_value (e.g. per second calculate mean average from each run)
    gdf = gdf[[extra_title_col, *data_cols, x_col, y_col]]
    gdf = gdf.groupby([extra_title_col, *data_cols, x_col]).aggregate(
        mean=pd.NamedAgg(y_col, np.mean),
        # If this is not a lambda, it will produce NaN for dataframes with only one value per bucket
        std=pd.NamedAgg(y_col, lambda x: np.std(x))
    )

    # Calculate data lines
    gdata = []
    if not gdf.empty:
        for data_tuple in unique_cartesian_product(df, extra_title_col, *data_cols):
            try:
                line_df = gdf.loc[data_tuple, ['mean', 'std']]
            except KeyError:
                # Combination in data_tuple does not exist
                continue
            if line_df.empty or line_df.isnull().values.all():
                # Combination in data_tuple has no data
                continue
            gdata.append((line_df, data_tuple))
    if len(gdata) == 0:
        return None
    gdata = sorted(gdata, key=lambda x: data_tuple_key_transform(x[1][1:]))

    # Merge line data into single df
    plot_df = pd.concat([x[0] for x in gdata], axis=1)
    # Make first category (named 0.0) start at the origin
    plot_df.iloc[0] = 0
    # Generate plot commands
    plot_cmds = []

    if plot_confidence_area:
        plot_cmds += [
            "using 1:((${y_col:d}+${std_col:d})/{y_div:f}):((${y_col:d}-${std_col:d})/{y_div:f})"
            " with filledcurve fillcolor rgb '#E0{fc:s}' fillstyle solid notitle"
            "".format(
                y_col=index * 2 + 2,
                std_col=index * 2 + 3,
                y_div=y_div,
                fc=get_line_color(line_map, (data_tuple[0], *tuple(data_tuple[i + 1] for i in line_color_indices)))
            )
            for index, (_, data_tuple) in enumerate(gdata)
        ]

    plot_cmds += [
        "using 1:(${y_col:d}/{y_div:f})"
        " with linespoints pointtype {pt:d} linecolor '#{lc:s}' title '{extra_title:s}{title:s}'"
        "".format(
            y_col=index * 2 + 2,
            y_div=y_div,
            pt=get_point_type(point_map, tuple(data_tuple[i + 1] for i in point_type_indices)),
            lc=get_line_color(line_map, (data_tuple[0], *tuple(data_tuple[i + 1] for i in line_color_indices))),
            extra_title=data_tuple[0] if len(data_tuple[0]) == 0 else ("%s " % data_tuple[0]),
            title=format_data_title(*data_tuple[1:])
        )
        for index, (_, data_tuple) in enumerate(gdata)
    ]

    return plot_df, plot_cmds, [data_tuple for _, data_tuple in gdata]


def plot_time_series_matrix(df: pd.DataFrame, out_dir: str, analysis_name: str, file_cols: List[str],
                            data_cols: List[str], matrix_x_cols: List[str], matrix_y_cols: List[str], x_col: str,
                            y_col: str, x_range: Optional[Tuple[int, int]], y_range: Optional[Tuple[int, int]],
                            x_bucket: Optional[float], y_div: float, x_label: str, y_label: str,
                            point_type_indices: List[int], line_color_indices: List[int],
                            format_data_title: Callable[[DataTuple], str],
                            format_subplot_title: Callable[[any, any], str],
                            format_file_title: Callable[[FileTuple], str],
                            format_file_base: Callable[[FileTuple], str],
                            data_sort_key: Callable[[DataTuple], any] = lambda x: x,
                            sort_matrix_x: Callable[[Iterable], Iterable] = lambda x: sorted(x),
                            sort_matrix_y: Callable[[Iterable], Iterable] = lambda y: sorted(y),
                            extra_title_col: Optional[str] = None) -> None:
    """
    Plot multiple time series graphs arranged like a 2d-matrix based on two data values. It is built for, but not
    restricted to having a time unit (e.g. seconds) on the x-axis of each individual graph.

    :param df: The dataframe to read the data from
    :param out_dir: Directory where all output files are placed
    :param analysis_name: A name for the analysis used in log statements
    :param file_cols: Column names that define values for which separate graphs are generated
    :param data_cols: Column names of the columns used for the data lines
    :param matrix_x_cols: Graphs are horizontally arranged based on values of these columns
    :param matrix_y_cols: Graphs are vertically arranged based on values of these columns
    :param x_col: Name of the column that has the data for the x-axis
    :param y_col: Name of the column that has the data for the y-axis
    :param x_range: (min, max) tuple for filtering the values for the x-axis
    :param y_range: (min, max) tuple for filtering the values for the y-axis
    :param x_bucket: Size of the bucket to use for aggregating data on the x-axis
    :param y_div: Number to divide the values on the y-axis by before plotting
    :param x_label: Label for the x-axis of the generated graphs
    :param y_label: LAbel for the y-axis of the generated graphs
    :param point_type_indices: Indices of file_cols used to determine point type
    :param line_color_indices: Indices of file_cols used to determine line color
    :param format_data_title: Function to format the title of a data line, receives a data_tuple (data_cols values)
    :param format_subplot_title: Function to format the title of a subplot, receives a tuple with the values of matrix_x_cols and matrix_y_cols
    :param format_file_title: Function to format the title of a graph, receives a file_tuple (file_cols values)
    :param format_file_base: Function to format the base name of a graph file, receives a file_tuple (file_cols values)
    :param data_sort_key: Function to transform a data tuple to a key to sort the data by
    :param sort_matrix_x: Function to sort values of the matrix_x_cols, graphs will be arranged accordingly
    :param sort_matrix_y: Function to sort values of the matrix_y_cols, graphs will be arranged accordingly
    :param extra_title_col: Name of the column that holds a string prefix for the data title
    """

    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map: PointMap = {}
    line_map: LineMap = {}

    if extra_title_col is None:
        extra_title_col = 'default_extra_title'
        df[extra_title_col] = ""

    for file_tuple in unique_cartesian_product(df, *file_cols):
        print_file_tuple = sprint_tuple(file_cols, file_tuple)
        logger.info("Generating %s matrix %s", analysis_name, print_file_tuple)
        file_df = filter_graph_data(df, x_col, x_range, file_cols, file_tuple)

        mx_unique = list(sort_matrix_x(unique_cartesian_product(file_df, *matrix_x_cols)))
        my_unique = list(sort_matrix_y(unique_cartesian_product(file_df, *matrix_y_cols)))
        mx_cnt = float(max(1, len(mx_unique)))
        my_cnt = float(max(1, len(my_unique)))
        sub_size = "%f, %f" % ((1.0 - MATRIX_KEY_SIZE) / mx_cnt, 1.0 / my_cnt)

        subfigures = []
        key_data = set()

        # [('1M,1M', '1M,1M,1M,1M', '1M,1M,1M,1M'), ('2M,2M', '1M,2M,2M,1M', '1M,2M,2M,1M'), ('4M,4M', '1M,4M,4M,1M', '1M,4M,4M,1M'), ('8M,8M', '1M,8M,8M,1M', '1M,8M,8M,1M')]
        # Generate subfigures
        y_max = max(1, np.ceil(file_df[y_col].replace(0, np.nan).quantile(.99) / y_div))
        for matrix_y_idx, matrix_y_tuple in enumerate(my_unique):
            for matrix_x_idx, matrix_x_tuple in enumerate(mx_unique):
                print_subplot_tuple = sprint_tuple([*file_cols, *matrix_x_cols, *matrix_y_cols],
                                                   (*file_tuple, *matrix_x_tuple, *matrix_y_tuple))
                prepared_data = prepare_time_series_graph_data(file_df,
                                                               x_col=x_col,
                                                               y_col=y_col,
                                                               x_range=x_range,
                                                               x_bucket=x_bucket,
                                                               y_div=y_div,
                                                               extra_title_col=extra_title_col,
                                                               file_cols=[*file_cols, *matrix_x_cols, *matrix_y_cols],
                                                               file_tuple=(
                                                                   *file_tuple, *matrix_x_tuple, *matrix_y_tuple),
                                                               data_cols=data_cols,
                                                               point_map=point_map,
                                                               line_map=line_map,
                                                               point_type_indices=point_type_indices,
                                                               line_color_indices=line_color_indices,
                                                               format_data_title=format_data_title,
                                                               data_tuple_key_transform=data_sort_key,
                                                               plot_confidence_area=True)
                if prepared_data is None:
                    logger.debug("No data for %s %s", analysis_name, print_subplot_tuple)
                    continue

                plot_df, plot_cmds, data_tuples = prepared_data

                # Add data for key
                key_data.update(data_tuples)

                subfigures.append(gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s"' % format_subplot_title(*matrix_x_tuple, *matrix_y_tuple),
                    key='off',
                    xlabel='"%s"' % x_label,
                    ylabel='"%s"' % y_label,
                    xrange=None if x_range is None else ('[%d:%d]' % x_range),
                    yrange='[0:]' % y_max if y_range is None else ('[%d:%d]' % y_range),
                    pointsize='0.5',
                    size=sub_size,
                    origin="%f, %f" % (matrix_x_idx * (1.0 - MATRIX_KEY_SIZE) / mx_cnt, matrix_y_idx / my_cnt)
                ))

        # Check if a matrix plot is useful
        if len(subfigures) <= 1:
            logger.debug("Skipping %s matrix plot for %s, not enough individual plots", analysis_name, print_file_tuple)
            continue

        # Add null plot for key
        key_cmds = [
            "NaN with linespoints pointtype %d linecolor '#%s' title '%s%s'" %
            (
                get_point_type(point_map, tuple(data_tuple[i + 1] for i in point_type_indices)),
                get_line_color(line_map, (data_tuple[0], *tuple(data_tuple[i + 1] for i in line_color_indices))),
                data_tuple[0] if len(data_tuple[0]) == 0 else ("%s " % data_tuple[0]),
                format_data_title(*data_tuple[1:])
            )
            for data_tuple in sorted(key_data, key=lambda dt: data_sort_key(dt[1:]))
        ]
        subfigures.append(gnuplot.make_plot(
            *key_cmds,
            key='on inside center center vertical Right samplen 2',
            pointsize='0.5',
            size="%f, 1" % MATRIX_KEY_SIZE,
            origin="%f, 0" % (1.0 - MATRIX_KEY_SIZE),
            title=None,
            xtics=None,
            ytics=None,
            xlabel=None,
            ylabel=None,
            xrange='[0:1]',
            yrange='[0:1]',
            border=None,
        ))

        gnuplot.multiplot(
            *subfigures,
            title='"%s"' % format_file_title(*file_tuple),
            term='pdf size %dcm, %dcm' %
                 (MATRIX_SUBPLOT_SIZE_CM[0] * mx_cnt, MATRIX_SUBPLOT_SIZE_CM[1] * my_cnt),
            output='"%s.pdf"' % os.path.join(out_dir, GRAPH_DIR, format_file_base(*file_tuple)),
        )


def analyze_opensand_goodput_pcaps_mp_cc_matrix_by_schedulers(df: pd.DataFrame, out_dir: str):
    for x_bucket in {1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_GOODPUT_%gS' % x_bucket,
                                file_cols=['mp_cc'],
                                data_cols=['protocol', 'pep'],
                                matrix_x_cols=['mp_sched', 'tbs', 'qbs', 'ubs'],
                                matrix_y_cols=['mp_cc'],
                                x_col='second',
                                y_col='bps',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                y_range=(0, GRAPH_Y_RANGE_GOODPUT),
                                x_bucket=x_bucket,
                                y_div=1,
                                x_label="Time (s)",
                                y_label="Goodput (mbps)",
                                point_type_indices=[],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep:
                                "%s%s" %
                                (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda mp_sched, tbs, qbs, ubs, mp_cc:
                                "Goodput Evolution - cc=%s - s=%s" % (mp_cc, sprint_scheduler_name(mp_sched)),
                                format_file_title=lambda mp_cc:
                                "Goodput Evolution - %s" % mp_cc,
                                format_file_base=lambda mp_cc:
                                "matrix_goodput_%gs_c%s" % (x_bucket, mp_cc),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: scheduler_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True, key=lambda yt: set(apply_si(yv) for yv in yt)))


def analyze_opensand_goodput_pcaps_mp_schedulers_matrix_by_cc(df: pd.DataFrame, out_dir: str):
    for x_bucket in {1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_GOODPUT_%gS' % x_bucket,
                                file_cols=['mp_sched'],
                                data_cols=['protocol', 'pep'],
                                matrix_x_cols=['mp_cc', 'tbs', 'qbs', 'ubs',],
                                matrix_y_cols=['mp_sched'],
                                x_col='second',
                                y_col='bps',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                y_range=(0, GRAPH_Y_RANGE_GOODPUT),
                                x_bucket=x_bucket,
                                y_div=1,
                                x_label="Time (s)",
                                y_label="Goodput (mbps)",
                                point_type_indices=[],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep:
                                "%s%s" %
                                (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda mp_cc, tbs, qbs, ubs, mp_sched:
                                "Goodput Evolution - s=%s - cc=%s" % (sprint_scheduler_name(mp_sched), mp_cc),
                                format_file_title=lambda mp_sched:
                                "Goodput Evolution - %s" % mp_sched,
                                format_file_base=lambda mp_sched:
                                "matrix_goodput_%gs_s%s" % (x_bucket, mp_sched),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: mp_cc_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True, key=lambda yt: set(apply_si(yv) for yv in yt)))


def analyze_opensand_rtt_pcap_mp_cc_matrix_by_schedulers(df: pd.DataFrame, out_dir: str):
    for x_bucket in {1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_RTT_%gS' % x_bucket,
                                file_cols=['mp_cc'],
                                data_cols=['protocol', 'pep'],
                                matrix_x_cols=['mp_sched', 'tbs', 'qbs', 'ubs'],
                                matrix_y_cols=['mp_cc'],
                                x_col='second',
                                y_col='avg_rtt',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                y_range=(0, GRAPH_Y_RANGE_RTT),
                                x_bucket=x_bucket,
                                y_div=1,
                                x_label="Time (s)",
                                y_label="RTT (ms)",
                                point_type_indices=[],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep:
                                "%s%s" %
                                (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda mp_sched, tbs, qbs, ubs, mp_cc:
                                "Average RTT - cc=%s - s=%s" % (mp_cc, sprint_scheduler_name(mp_sched)),
                                format_file_title=lambda mp_cc:
                                "Average RTT - %s" % mp_cc,
                                format_file_base=lambda mp_cc:
                                "matrix_packet_rtt_pcap_%gs_c%s" % (x_bucket, mp_cc),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: scheduler_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_rtt_pcap_mp_schedulers_matrix_by_cc(df: pd.DataFrame, out_dir: str):
    for x_bucket in {1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_RTT_%gS' % x_bucket,
                                file_cols=['mp_sched'],
                                data_cols=['protocol', 'pep'],
                                matrix_x_cols=['mp_cc', 'tbs', 'qbs', 'ubs'],
                                matrix_y_cols=['mp_sched'],
                                x_col='second',
                                y_col='avg_rtt',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                y_range=(0, GRAPH_Y_RANGE_RTT),
                                x_bucket=x_bucket,
                                y_div=1,
                                x_label="Time (s)",
                                y_label="RTT (ms)",
                                point_type_indices=[],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep:
                                "%s%s" %
                                (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda mp_cc, tbs, qbs, ubs, mp_sched:
                                "Average RTT - s=%s - cc=%s " % (sprint_scheduler_name(mp_sched), mp_cc),
                                format_file_title=lambda mp_sched:
                                "Average RTT - %s" % mp_sched,
                                format_file_base=lambda mp_sched:
                                "matrix_packet_rtt_pcap_%gs_s%s" % (x_bucket, mp_sched),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: mp_cc_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_retransmits_mp_cc_matrix_by_schedulers(df: pd.DataFrame, out_dir: str):
    for x_bucket in {1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_PACKET_LOSS_%gS' % x_bucket,
                                file_cols=['mp_cc'],
                                data_cols=['protocol', 'pep'],
                                matrix_x_cols=['mp_sched', 'tbs', 'qbs', 'ubs'],
                                matrix_y_cols=['mp_cc'],
                                x_col='second',
                                y_col='retransmits',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                y_range=(0, GRAPH_Y_RANGE_RETRANSMITS),
                                x_bucket=x_bucket,
                                y_div=1,
                                x_label="Time (s)",
                                y_label="Retransmission rate",
                                point_type_indices=[],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep:
                                "%s%s" %
                                (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda mp_sched, tbs, qbs, ubs, mp_cc:
                                "Packet Loss - cc=%s - s=%s" % (mp_cc, sprint_scheduler_name(mp_sched)),
                                format_file_title=lambda mp_cc:
                                "Packet Loss - %s" % mp_cc,
                                format_file_base=lambda mp_cc:
                                "matrix_packet_loss_%gs_c%s" % (x_bucket, mp_cc),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: scheduler_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_retransmits_mp_schedulers_matrix_by_cc(df: pd.DataFrame, out_dir: str):
    for x_bucket in {1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_PACKET_LOSS_%gS' % x_bucket,
                                file_cols=['mp_sched'],
                                data_cols=['protocol', 'pep'],
                                matrix_x_cols=['mp_cc', 'tbs', 'qbs', 'ubs'],
                                matrix_y_cols=['mp_sched'],
                                x_col='second',
                                y_col='retransmits',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                y_range=(0, GRAPH_Y_RANGE_RETRANSMITS),
                                x_bucket=x_bucket,
                                y_div=1,
                                x_label="Time (s)",
                                y_label="Retransmission rate",
                                point_type_indices=[],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep:
                                "%s%s" %
                                (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda mp_cc, tbs, qbs, ubs, mp_sched:
                                "Packet Loss - s=%s - cc=%s" % (sprint_scheduler_name(mp_sched), mp_cc),
                                format_file_title=lambda mp_sched:
                                "Packet Loss - %s" % mp_sched,
                                format_file_base=lambda mp_sched:
                                "matrix_packet_loss_%gs_s%s" % (x_bucket, mp_sched),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: mp_cc_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def __analyze_all_goodput(parsed_results: dict, measure_type: MeasureType, out_dir: str,
                          time_series_cols: List[str]) -> None:
    logger.info("Analyzing MPTCP UL (s2c) goodput from PCAPs")
    # return
    goodput_cols = [*time_series_cols, 'bps']
    df_goodput = pd.concat([
        parsed_results['mptcp_gp_dl'][goodput_cols],
        parsed_results['mptcp_gp_ul'][goodput_cols],
        parsed_results['tcp_gp_ul_lte'][goodput_cols],
        parsed_results['tcp_gp_ul_sat'][goodput_cols],
        parsed_results['tcp_gp_dl_lte'][goodput_cols],
        parsed_results['tcp_gp_dl_sat'][goodput_cols],
    ], axis=0, ignore_index=True)

    if measure_type == MeasureType.NETEM:
        pass
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_goodput_pcaps_mp_cc_matrix_by_schedulers(df_goodput, out_dir)
        analyze_opensand_goodput_pcaps_mp_schedulers_matrix_by_cc(df_goodput, out_dir)


def __analyze_all_rtt(parsed_results: dict, measure_type: MeasureType, out_dir: str,
                              time_series_cols: List[str]) -> None:
    logger.info("Analyzing MPTCP UL (s2c) RTTs from PCAPs")
    # return

    rtt_cols = [*time_series_cols, 'avg_rtt']

    df_rtt = pd.concat([
        parsed_results['mptcp_rtt_dl'][rtt_cols],
        parsed_results['mptcp_rtt_ul'][rtt_cols],
        parsed_results['tcp_rtt_ul_lte'][rtt_cols],
        parsed_results['tcp_rtt_ul_sat'][rtt_cols],
        parsed_results['tcp_rtt_dl_lte'][rtt_cols],
        parsed_results['tcp_rtt_dl_sat'][rtt_cols],
    ], axis=0, ignore_index=True)

    if measure_type == MeasureType.NETEM:
        pass
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_rtt_pcap_mp_cc_matrix_by_schedulers(df_rtt, out_dir)
        analyze_opensand_rtt_pcap_mp_schedulers_matrix_by_cc(df_rtt, out_dir)
        pass


def __analyze_all_packet_loss(parsed_results: dict, measure_type: MeasureType, out_dir: str,
                              time_series_cols: List[str]) -> None:
    logger.info("Analyzing MPTCP UL (s2c) packet loss from PCAPs")

    retransmit_cols = [*time_series_cols, 'retransmission', 'retransmission_rate', 'loss', 'loss_rate']
    df_pkt_loss = pd.concat([
        parsed_results['mptcp_loss_dl'][retransmit_cols],
        parsed_results['mptcp_loss_ul'][retransmit_cols],
        parsed_results['tcp_loss_ul_lte'][retransmit_cols],
        parsed_results['tcp_loss_ul_sat'][retransmit_cols],
        parsed_results['tcp_loss_dl_lte'][retransmit_cols],
        parsed_results['tcp_loss_dl_sat'][retransmit_cols],
    ], axis=0, ignore_index=True)

    if measure_type == MeasureType.NETEM:
        pass
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_retransmits_mp_cc_matrix_by_schedulers(df_pkt_loss, out_dir)
        analyze_opensand_retransmits_mp_schedulers_matrix_by_cc(df_pkt_loss, out_dir)
        pass


def __analyze_all_stats(parsed_results: dict, measure_type: MeasureType, out_dir: str) -> None:
    logger.info("Analyzing stats")
    return
    if measure_type == MeasureType.OPENSAND:
        df_stats = pd.DataFrame(parsed_results['stats'])
        df_runs = pd.DataFrame(parsed_results['runs'])
        df_stats.index = df_stats.index.total_seconds()
        df_runs.index = df_runs.index.total_seconds()
        analyze_stats(df_stats, df_runs, out_dir)


def analyze_all(parsed_results: dict, measure_type: MeasureType, out_dir: str, multi_process: bool = False):
    time_series_cols = ['protocol', 'pep', 'sat', 'run', 'second']
    if measure_type == MeasureType.NETEM:
        time_series_cols.extend(['rate', 'loss', 'queue'])
    elif measure_type == MeasureType.OPENSAND:
        time_series_cols.extend(['attenuation', 'ccs', 'tbs', 'qbs', 'ubs', 'mp_sched', 'mp_cc'])

    if multi_process:
        processes = [
            mp.Process(target=__analyze_all_goodput, name='goodput',
                       args=(parsed_results, measure_type, out_dir, time_series_cols)),
            mp.Process(target=__analyze_all_rtt, name='rtt',
                       args=(parsed_results, measure_type, out_dir)),
            mp.Process(target=__analyze_all_packet_loss, name='retransmits',
                       args=(parsed_results, measure_type, out_dir, time_series_cols)),
        ]

        for p in processes:
            p.start()

        __analyze_all_stats(parsed_results, measure_type, out_dir)

        for p in processes:
            p.join()
    else:
        __analyze_all_goodput(parsed_results, measure_type, out_dir, time_series_cols)
        __analyze_all_rtt(parsed_results, measure_type, out_dir, time_series_cols)
        # __analyze_all_packet_loss(parsed_results, measure_type, out_dir, time_series_cols)
        __analyze_all_stats(parsed_results, measure_type, out_dir)
