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

import csv
import json
import multiprocessing as mp
import os
import os.path
import re
import statistics
import subprocess
import sys
from audioop import avg
from code import interact
from collections import defaultdict
from datetime import datetime
from itertools import islice
from multiprocessing.dummy.connection import Connection
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

import common
from common import logger


def weird_division(n, d):
    return n / d if d else 0


def parse_auto_detect(in_dir: str, out_dir: str) -> Dict:
    auto_detect = {}
    try:
        with open(os.path.join(in_dir, "environment.txt"), 'r') as env_file:
            auto_detect = {key: value for key, value in
                           filter(lambda x: len(x) == 2 and x[0] in ('MEASURE_TIME', 'REPORT_INTERVAL'),
                                  [line.split('=', 1) for line in env_file.readlines()])
                           }
    except IOError:
        pass

    with open(os.path.join(out_dir, common.AUTO_DETECT_FILE), 'w+') as out_file:
        out_file.writelines(["%s=%s" % (key, str(value))
                            for key, value in auto_detect.items()])

    return auto_detect


def bps_factor(prefix: str):
    factor = {'K': 10 ** 3, 'M': 10 ** 6, 'G': 10 ** 9, 'T': 10 ** 12, 'P': 10 ** 15, 'E': 10 ** 18, 'Z': 10 ** 21,
              'Y': 10 ** 24}
    prefix = prefix.upper()
    return factor[prefix] if prefix in factor else 1


def detect_measure_type(in_dir: str, out_dir: str) -> common.MeasureType:
    logger.info("Detecting type of measurement")
    measure_type = None
    is_certain = True

    path = os.path.join(in_dir, "opensand.log")
    if os.path.isfile(path):
        if measure_type is not None:
            is_certain = False
        measure_type = common.MeasureType.OPENSAND

    path = os.path.join(in_dir, "measure.log")
    if os.path.isfile(path):
        if measure_type is not None:
            is_certain = False
        measure_type = common.MeasureType.NETEM

    if measure_type is None or not is_certain:
        logger.error("Failed to detect measurement type!")
        sys.exit(4)

    logger.info("Measure type: %s", measure_type.name)
    with open(os.path.join(out_dir, common.TYPE_FILE), 'w+') as type_file:
        type_file.write(measure_type.name)

    return measure_type


def __read_config_from_scenario(in_dir: str, scenario_name: str) -> Dict:
    config = {
        'name': scenario_name
    }

    with open(os.path.join(in_dir, scenario_name, 'config.txt'), 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    return config


def list_result_folders(root_folder: str) -> Generator[str, None, None]:
    for folder_name in os.listdir(root_folder):
        path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(path):
            logger.debug("'%s' is not a directory, skipping", folder_name)
            continue
        yield folder_name


def extend_df(df: pd.DataFrame, by: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Extends the dataframe containing the data of a single file (by) with the information given in the kwargs so that it
    can be appended to the main dataframe (df)
    :param df: The main dataframe
    :param by: The dataframe to extend by
    :param kwargs: Values to use for new columns in by
    :return: The extended df
    """

    aliases = {
        'sat': ['delay', 'orbit'],
        'queue': ['queue_overhead_factor'],
    }
    missing_cols = set(df.columns).difference(set(by.columns))
    for col_name in missing_cols:
        col_value = np.nan

        if col_name in kwargs:
            col_value = kwargs[col_name]
        elif col_name in aliases:
            for alias_col in aliases[col_name]:
                if alias_col in kwargs:
                    col_value = kwargs[alias_col]
                    break

        by[col_name] = col_value
    return df.append(by, ignore_index=True)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the data types of the columns in a data frame.
    :param df: The dataframe to fix
    :return:
    """

    # Cleanup values
    if 'rate' in df:
        df['rate'] = df['rate'].apply(
            lambda x: np.nan if str(x) == 'nan' else ''.join(c for c in str(x) if c.isdigit() or c == '.'))
    if 'loss' in df:
        df['loss'] = df['loss'].apply(
            lambda x: np.nan if str(x) == 'nan' else float(''.join(c for c in str(x) if c.isdigit() or c == '.')) / 100)

    defaults = {
        np.int32: -1,
        np.str: "",
        np.bool: False,
    }
    dtypes = {
        'protocol': np.str,
        'pep': np.bool,
        'sat': np.str,
        'rate': np.int32,
        'loss': float,
        'queue': np.int32,
        'run': np.int32,
        'second': np.float32,
        'bps': np.float64,
        'bytes': np.int32,
        'packets_received': np.int32,
        'cwnd': np.int32,
        'packets_sent': np.int32,
        'packets_lost': np.int32,
        'con_est': np.float64,
        'ttfb': np.float64,
        'omitted': np.bool,
        'rtt': np.float32,
        'seq': np.int32,
        'ttl': np.int32,
        'rtt_min': np.float32,
        'rtt_avg': np.float32,
        'rtt_max': np.float32,
        'rtt_mdev': np.float32,
        'name': np.str,
        'cpu_load': np.float32,
        'ram_usage': np.float32,
        'attenuation': np.int32,
        'tbs': np.str,
        'qbs': np.str,
        'ubs': np.str,
        'prime': np.float32,
        'retransmits': np.float32,
        'dup_acks': np.int32,
        'lost_segments': np.int32,
        'fast_retransmits': np.int32,
        'ofo_segments': np.int32,
        'avg_rtt': np.float32,
        'path_util_lte': np.float32,
        'path_util_sat': np.float32,
        'ofo_queue_size': np.float32,
    }

    # Set defaults
    df = df.fillna(
        {col: defaults.get(dtypes[col], np.nan) for col in dtypes.keys()})

    cols = set(df.columns).intersection(dtypes.keys())
    return df.astype({col_name: dtypes[col_name] for col_name in cols})


def __mp_function_wrapper(parse_func: Callable[..., any], conn: Connection, *args, **kwargs) -> None:
    result = parse_func(*args, **kwargs)
    conn.send(result)
    conn.close()


def __parse_slice(parse_func: Callable[..., pd.DataFrame], in_dir: str, compression: int, run_type: int, scenarios: List[Tuple[str, Dict]],
                  df_cols: List[str], protocol: str, entity: str) -> pd.DataFrame:
    """
    Parse a slice of the protocol entity results using the given function.
    :param parse_func: The function to parse a single scenario.
    :param in_dir: The directory containing the measurement results.
    :param compression: Wether to perform decompression before processing or on-the-fly
    :param run_type: The run type of the measurement (SPTCP or MPTCP duplex)
    :param scenarios: The scenarios to parse within the in_dir.
    :param df_cols: The column names for columns in the resulting dataframe.
    :param protocol: The name of the protocol that is being parsed.
    :param entity: Then name of the entity that is being parsed.
    :return: A dataframe containing the combined results of the specified scenarios.
    """

    df_slice = pd.DataFrame(columns=df_cols)

    for folder, config in scenarios:
        for pep in (False, True):
            df = parse_func(in_dir, compression, run_type, folder, pep=pep)
            if df is not None:
                df_slice = extend_df(
                    df_slice, df, protocol=protocol, pep=pep, **config)
            else:
                logger.warning("No data %s%s %s data in %s", protocol,
                               " (pep)" if pep else "", entity, folder)

    return df_slice


def __mp_parse_slices(num_procs: int, parse_func: Callable[..., pd.DataFrame], in_dir: str, compression: int, run_type: int,
                      scenarios: Dict[str, Dict], df_cols: List[str], protocol: str, entity: str) -> pd.DataFrame:
    """
    Parse all protocol entity results using the given function in multiple processes.
    :param num_procs: The number of processes to spawn.
    :param parse_func: The function to parse a single scenario.
    :param in_dir: The directory containing the measurement results.
    :param compression: Wether to perform decompression before processing or on-the-fly
    :param run_type: The run type of the measurement (SPTCP or MPTCP duplex)
    :param scenarios: The scenarios to parse within the in_dir.
    :param df_cols: The column names for columns in the resulting dataframe.
    :param protocol: The name of the protocol that is being parsed.
    :param entity: Then name of the entity that is being parsed.
    :return:
    """

    tasks = [
        (
            "%s_%s_%d" % (protocol, entity, i),
            list(islice(scenarios.items(), i, sys.maxsize, num_procs)),
            mp.Pipe()
        )
        for i in range(num_procs)
    ]
    processes = [
        mp.Process(target=__mp_function_wrapper, name=name,
                   args=(__parse_slice, child_con, parse_func, in_dir, s_slice, df_cols, protocol, entity))
        for name, s_slice, (_, child_con) in tasks
    ]

    # Start processes
    for p in processes:
        p.start()

    # Collect results
    slice_dfs = [
        parent_con.recv()
        for _, _, (parent_con, _) in tasks
    ]

    # Wait for processes to finish
    for p in processes:
        p.join()

    return pd.concat(slice_dfs, axis=0, ignore_index=True)


def parse_mptcp_goodput_dl(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                           multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP DL goodput values from PCAPs
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP DL goodput results")

    df_cols = [*config_cols, 'run', 'second', 'bps']
    if multi_process:
        df_mptcp_gp_dl = __mp_parse_slices(4, __parse_mptcp_goodput_dl_from_scenario, in_dir, compression, run_type, scenarios,
                                           df_cols, 'mptcp-dl', 'gp_dl')
    else:
        df_mptcp_gp_dl = __parse_slice(__parse_mptcp_goodput_dl_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                       df_cols, 'mptcp-dl', 'gp_dl')

    logger.debug("Fixing MPTCP DL goodput data types")
    df_mptcp_gp_dl = fix_dtypes(df_mptcp_gp_dl)

    logger.info("Saving MPTCP DL goodput data")
    df_mptcp_gp_dl.to_pickle(os.path.join(out_dir, 'mptcp_gp_dl.pkl'))
    with open(os.path.join(out_dir, 'mptcp_gp_dl.csv'), 'w+') as out_file:
        df_mptcp_gp_dl.to_csv(out_file)

    return df_mptcp_gp_dl


def __parse_mptcp_goodput_dl_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP DL goodput results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s DL goodput files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'bps'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue
        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_gput_1.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -G 1000 -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path + " -G 1000 -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            initial_ts = first_row[0]
            goodput_per_sec = defaultdict(list)
            # for interval in results:
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                duration = float(curr_ts) - float(initial_ts)
                duration = int(duration)
                goodput = float(interval[1])
                goodput_per_sec[duration].append(goodput)

        for duration in goodput_per_sec:
            df = df.append({
                'run': run,
                'second': duration,
                'bps': statistics.mean(goodput_per_sec[duration]) * 8,
            }, ignore_index=True)

        # Remove intermediate files
        # files = os.listdir(path)
        # for file in files:
        #     if file.endswith(".csv"):
        #         os.remove(os.path.join(path, file))

    if df.empty:
        logger.warning("%s: No MPTCP%s DL goodput found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_goodput_ul(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                           multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP UL goodput values from PCAPs
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP UL goodput results")

    df_cols = [*config_cols, 'run', 'second', 'bps']
    if multi_process:
        df_mptcp_gp_ul = __mp_parse_slices(4, __parse_mptcp_goodput_ul_from_scenario, in_dir, compression, run_type, scenarios,
                                           df_cols, 'mptcp-ul', 'gp_ul')
    else:
        df_mptcp_gp_ul = __parse_slice(__parse_mptcp_goodput_ul_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                       df_cols, 'mptcp-ul', 'gp_ul')

    logger.debug("Fixing MPTCP UL goodput data types")
    df_mptcp_gp_ul = fix_dtypes(df_mptcp_gp_ul)

    logger.info("Saving MPTCP UL goodput data")
    df_mptcp_gp_ul.to_pickle(os.path.join(out_dir, 'mptcp_gp_ul.pkl'))
    with open(os.path.join(out_dir, 'mptcp_gp_ul.csv'), 'w+') as out_file:
        df_mptcp_gp_ul.to_csv(out_file)

    return df_mptcp_gp_ul


def __parse_mptcp_goodput_ul_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP UL goodput results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s UL goodput files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'bps'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue
        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_gput_1.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -G 1000 -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path + " -G 1000 -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            initial_ts = first_row[0]
            goodput_per_sec = defaultdict(list)
            # for interval in results:
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                duration = float(curr_ts) - float(initial_ts)
                duration = int(duration)
                goodput = float(interval[1])
                goodput_per_sec[duration].append(goodput)

        for duration in goodput_per_sec:
            df = df.append({
                'run': run,
                'second': duration,
                'bps': statistics.mean(goodput_per_sec[duration]) * 8,
            }, ignore_index=True)

        # Remove intermediate files
        # files = os.listdir(path)
        # for file in files:
        #     if file.endswith(".csv"):
        #         os.remove(os.path.join(path, file))

    if df.empty:
        logger.warning("%s: No MPTCP%s UL goodput found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_goodput_ul_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                             multi_process: bool = False) -> pd.DataFrame:
    """
    Parse TCP UL LTE goodput values from PCAPs
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing TCP UL LTE goodput results")

    df_cols = [*config_cols, 'run', 'second', 'bps']
    if multi_process:
        df_tcp_gp_ul_lte = __mp_parse_slices(4, __parse_tcp_goodput_ul_lte_from_scenario, in_dir, compression, run_type, scenarios,
                                             df_cols, 'tcp-ul-lte', 'gp_ul_lte')
    else:
        df_tcp_gp_ul_lte = __parse_slice(__parse_tcp_goodput_ul_lte_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                         df_cols, 'tcp-ul-lte', 'gp_ul_lte')

    logger.debug("Fixing TCP UL LTE goodput data types")
    df_tcp_gp_ul_lte = fix_dtypes(df_tcp_gp_ul_lte)

    logger.info("Saving TCP UL LTE goodput data")
    df_tcp_gp_ul_lte.to_pickle(os.path.join(out_dir, 'tcp_gp_ul_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_gp_ul_lte.csv'), 'w+') as out_file:
        df_tcp_gp_ul_lte.to_csv(out_file)

    return df_tcp_gp_ul_lte


def __parse_tcp_goodput_ul_lte_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the TCP UL LTE goodput results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing TCP%s UL LTE goodput files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'bps'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue
        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # tshark intermediate file name
        processed_file_name = None

        if int(compression) == 1:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -q -z io,stat,1,\"BYTES()ip.src==" + common.CLIENT_IP_LTE + \
                "\" > " + processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -q -z io,stat,1,\"BYTES()ip.src==" + common.CLIENT_IP_LTE + \
                "\" > " + processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 12, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                bps = int(interval[2])/125000
                df = df.append({
                    'run': run,
                    'second': int(duration),
                    'bps': bps,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No TCP%s UL LTE goodput found",
                       scenario_name, " (pep)" if pep else "")
    # else:
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #         print(df)

    return df


def parse_tcp_goodput_ul_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                             multi_process: bool = False) -> pd.DataFrame:
    """
    Parse TCP UL SAT goodput values from PCAPs
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP UL SAT goodput results")

    df_cols = [*config_cols, 'run', 'second', 'bps']
    if multi_process:
        df_tcp_gp_ul_sat = __mp_parse_slices(4, __parse_tcp_goodput_ul_sat_from_scenario, in_dir, compression, run_type, scenarios,
                                             df_cols, 'tcp-ul-sat', 'gp_ul_sat')
    else:
        df_tcp_gp_ul_sat = __parse_slice(__parse_tcp_goodput_ul_sat_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                         df_cols, 'tcp-ul-sat', 'gp_ul_sat')

    logger.debug("Fixing TCP UL SAT goodput data types")
    df_tcp_gp_ul_sat = fix_dtypes(df_tcp_gp_ul_sat)

    logger.info("Saving TCP UL SAT goodput data")
    df_tcp_gp_ul_sat.to_pickle(os.path.join(out_dir, 'tcp_gp_ul_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_gp_ul_sat.csv'), 'w+') as out_file:
        df_tcp_gp_ul_sat.to_csv(out_file)

    return df_tcp_gp_ul_sat


def __parse_tcp_goodput_ul_sat_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the TCP UL SAT goodput results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing TCP%s UL SAT goodput files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'bps'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # tshark intermediate file name
        processed_file_name = None

        if int(compression) == 1:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -q -z io,stat,1,\"BYTES()ip.src==" + common.CLIENT_IP_SAT + \
                "\" > " + processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -q -z io,stat,1,\"BYTES()ip.src==" + common.CLIENT_IP_SAT + \
                "\" > " + processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 12, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                bps = int(interval[2])/125000
                df = df.append({
                    'run': run,
                    'second': int(duration),
                    'bps': bps,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No TCP%s UL SAT goodput found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_goodput_dl_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                             multi_process: bool = False) -> pd.DataFrame:
    """
    Parse TCP DL LTE goodput values from PCAPs
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing TCP DL LTE goodput results")

    df_cols = [*config_cols, 'run', 'second', 'bps']
    if multi_process:
        df_tcp_gp_dl_lte = __mp_parse_slices(4, __parse_tcp_goodput_dl_lte_from_scenario, in_dir, compression, run_type, scenarios,
                                             df_cols, 'tcp-dl-lte', 'gp_dl_lte')
    else:
        df_tcp_gp_dl_lte = __parse_slice(__parse_tcp_goodput_dl_lte_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                         df_cols, 'tcp-dl-lte', 'gp_dl_lte')

    logger.debug("Fixing TCP DL LTE goodput data types")
    df_tcp_gp_dl_lte = fix_dtypes(df_tcp_gp_dl_lte)

    logger.info("Saving TCP DL LTE goodput data")
    df_tcp_gp_dl_lte.to_pickle(os.path.join(out_dir, 'tcp_gp_dl_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_gp_dl_lte.csv'), 'w+') as out_file:
        df_tcp_gp_dl_lte.to_csv(out_file)

    return df_tcp_gp_dl_lte


def __parse_tcp_goodput_dl_lte_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the TCP DL LTE goodput results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing TCP%s DL LTE goodput files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'bps'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # tshark intermediate file name
        processed_file_name = None

        if int(compression) == 1:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP + \
                " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + "\" > " + \
                processed_file_name if int(run_type) == 4 \
                else "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP_MP + \
                " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + "\" > " + \
                processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP + \
                " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + "\" > " + \
                processed_file_name if int(run_type) == 4 \
                else "tshark -r " + file_path + " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP_MP + \
                " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + "\" > " + \
                processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 12, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                bps = int(interval[2])/125000
                df = df.append({
                    'run': run,
                    'second': int(duration),
                    'bps': bps,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No TCP%s DL LTE goodput found",
                       scenario_name, " (pep)" if pep else "")
    # else:
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #         print(df)

    return df


def parse_tcp_goodput_dl_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                             multi_process: bool = False) -> pd.DataFrame:
    """
    Parse TCP DL SAT goodput values from PCAPs
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing TCP DL SAT goodput results")

    df_cols = [*config_cols, 'run', 'second', 'bps']
    if multi_process:
        df_tcp_gp_dl_sat = __mp_parse_slices(4, __parse_tcp_goodput_dl_sat_from_scenario, in_dir, compression, run_type, scenarios,
                                             df_cols, 'tcp-dl-sat', 'gp_dl_sat')
    else:
        df_tcp_gp_dl_sat = __parse_slice(__parse_tcp_goodput_dl_sat_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                         df_cols, 'tcp-dl-sat', 'gp_dl_sat')

    logger.debug("Fixing TCP DL SAT goodput data types")
    df_tcp_gp_dl_sat = fix_dtypes(df_tcp_gp_dl_sat)

    logger.info("Saving TCP DL SAT goodput data")
    df_tcp_gp_dl_sat.to_pickle(os.path.join(out_dir, 'tcp_gp_dl_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_gp_dl_sat.csv'), 'w+') as out_file:
        df_tcp_gp_dl_sat.to_csv(out_file)

    return df_tcp_gp_dl_sat


def __parse_tcp_goodput_dl_sat_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the TCP DL SAT goodput results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing TCP%s DL SAT goodput files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'bps'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # tshark intermediate file name
        processed_file_name = None

        if int(compression) == 1:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP + \
                " and ip.dst==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + "\" > " + \
                processed_file_name if int(run_type) == 3 \
                else "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP_MP + \
                " and ip.dst==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + "\" > " + \
                processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP + \
                " and ip.dst==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + "\" > " + \
                processed_file_name if int(run_type) == 3 else \
                "tshark -r " + file_path + " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP_MP + \
                " and ip.dst==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + "\" > " + \
                processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 12, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                bps = int(interval[2])/125000
                df = df.append({
                    'run': run,
                    'second': int(duration),
                    'bps': bps,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No TCP%s DL SAT (c2s) goodput found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_rtt_dl(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                       multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP DL RTT values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP DL RTT results")

    df_cols = [*config_cols, 'run', 'second', 'avg_rtt']
    if multi_process:
        df_mptcp_rtt_dl = __mp_parse_slices(4, __parse_mptcp_rtt_dl_from_pcaps, in_dir, compression, run_type, scenarios,
                                            df_cols, 'mptcp-dl', 'rtt_dl')
    else:
        df_mptcp_rtt_dl = __parse_slice(__parse_mptcp_rtt_dl_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                        df_cols, 'mptcp-dl', 'rtt_dl')

    logger.debug("Fixing MPTCP DL RTT data types")
    df_mptcp_rtt_dl = fix_dtypes(df_mptcp_rtt_dl)

    logger.info("Saving MPTCP DL RTT data")
    df_mptcp_rtt_dl.to_pickle(os.path.join(out_dir, 'mptcp_rtt_dl.pkl'))
    with open(os.path.join(out_dir, 'mptcp_rtt_dl.csv'), 'w+') as out_file:
        df_mptcp_rtt_dl.to_csv(out_file)

    return df_mptcp_rtt_dl


def parse_mptcp_path_util_ul(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                             multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP UL path utilization from PCAPs
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP UL path utilization results")

    df_cols = [*config_cols, 'run', 'second', 'path_util_lte', 'path_util_sat']
    if multi_process:
        df_mptcp_path_util_ul = __mp_parse_slices(4, __parse_mptcp_path_util_ul_from_scenario, in_dir, compression, run_type, scenarios,
                                                  df_cols, 'path-util-ul', 'path_util_ul')
    else:
        df_mptcp_path_util_ul = __parse_slice(__parse_mptcp_path_util_ul_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                              df_cols, 'path-util-ul', 'path_util_ul')

    logger.debug("Fixing MPTCP UL path utilization data types")
    df_mptcp_path_util_ul = fix_dtypes(df_mptcp_path_util_ul)

    logger.info("Saving MPTCP UL path utilization  data")
    df_mptcp_path_util_ul.to_pickle(
        os.path.join(out_dir, 'mptcp_path_util_ul.pkl'))
    with open(os.path.join(out_dir, 'mptcp_path_util_ul.csv'), 'w+') as out_file:
        df_mptcp_path_util_ul.to_csv(out_file)

    return df_mptcp_path_util_ul


def __parse_mptcp_path_util_ul_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP UL path utilization results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s UL path utilization files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(
        columns=['run', 'second', 'path_util_lte', 'path_util_sat'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue
        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # tshark intermediate file name
        processed_file_name = None

        if int(compression) == 1:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -q -z io,stat,1,\"BYTES()ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"BYTES()ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.GSTREAMER_PORT + "\" > " + processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -q -z io,stat,1,\"BYTES()ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"BYTES()ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.GSTREAMER_PORT + "\" > " + processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 13, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                try:
                    path_util_lte = float(interval[2])
                    path_util_sat = float(interval[3])
                    total_path_util = path_util_lte + path_util_sat
                    path_util_ratio_lte = path_util_lte / total_path_util
                    path_util_ratio_sat = path_util_sat / total_path_util
                except:
                    pass
                df = df.append({
                    'run': run,
                    'second': int(duration),
                    'path_util_lte': path_util_ratio_lte,
                    'path_util_sat': path_util_ratio_sat,
                }, ignore_index=True)

        # Remove intermediate files
        # remove_cmd = "rm " + processed_file_name
        # os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No MPTCP%s UL path utilization found",
                       scenario_name, " (pep)" if pep else "")
    # else:
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #         print(df)

    return df


def parse_mptcp_path_util_dl(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                             multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP DL path utilization from PCAPs
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP DL path utilization results")

    df_cols = [*config_cols, 'run', 'second', 'path_util_lte', 'path_util_sat']
    if multi_process:
        df_mptcp_path_util_dl = __mp_parse_slices(4, __parse_mptcp_path_util_dl_from_scenario, in_dir, compression, run_type, scenarios,
                                                  df_cols, 'path-util-dl', 'path_util_dl')
    else:
        df_mptcp_path_util_dl = __parse_slice(__parse_mptcp_path_util_dl_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                              df_cols, 'path-util-dl', 'path_util_dl')

    logger.debug("Fixing MPTCP DL path utilization data types")
    df_mptcp_path_util_dl = fix_dtypes(df_mptcp_path_util_dl)

    logger.info("Saving MPTCP DL path utilization  data")
    df_mptcp_path_util_dl.to_pickle(
        os.path.join(out_dir, 'mptcp_path_util_dl.pkl'))
    with open(os.path.join(out_dir, 'mptcp_path_util_dl.csv'), 'w+') as out_file:
        df_mptcp_path_util_dl.to_csv(out_file)

    return df_mptcp_path_util_dl


def __parse_mptcp_path_util_dl_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP DL path utilization results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s DL path utilization files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(
        columns=['run', 'second', 'path_util_lte', 'path_util_sat'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue
        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # tshark intermediate file name
        processed_file_name = None

        if int(compression) == 1:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP_MP + " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"BYTES()ip.src==" + common.SERVER_IP_MP + " and ip.dst==" + common.CLIENT_IP_SAT + \
                " and tcp.port==" + common.IPERF_PORT + "\" > " + processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # tshark intermediate file name
            processed_file_name = path + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -q -z io,stat,1,\"BYTES()ip.src==" + common.SERVER_IP_MP + " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"BYTES()ip.src==" + common.SERVER_IP_MP + " and ip.dst==" + common.CLIENT_IP_SAT + \
                " and tcp.port==" + common.IPERF_PORT + "\" > " + processed_file_name
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 13, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                try:
                    path_util_lte = float(interval[2])
                    path_util_sat = float(interval[3])
                    total_path_util = path_util_lte + path_util_sat
                    path_util_ratio_lte = path_util_lte / total_path_util
                    path_util_ratio_sat = path_util_sat / total_path_util
                except:
                    pass
                df = df.append({
                    'run': run,
                    'second': int(duration),
                    'path_util_lte': path_util_ratio_lte,
                    'path_util_sat': path_util_ratio_sat,
                }, ignore_index=True)

        # Remove intermediate files
        # remove_cmd = "rm " + processed_file_name
        # os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No MPTCP%s DL path utilization found",
                       scenario_name, " (pep)" if pep else "")
    # else:
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #         print(df)

    return df


def __parse_mptcp_rtt_dl_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP DL RTT results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s DL RTT files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_rtt'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue
        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_rtt_seq_ts_0.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -r 2 -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -r 2 -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        df_i = pd.read_csv(processed_file_name, header=None)
        df_i.iloc[:, 0] = df_i.iloc[:, 0] - df_i.iloc[0, 0]

        df['second'] = df_i.iloc[:, 0]
        df['avg_rtt'] = df_i.iloc[:, 1]
        df = df.assign(run=run)

        # with open(processed_file_name, 'r') as file:
        #     results = csv.reader(file, delimiter=',')
        #     if results is None:
        #         logger.warning("%s: '%s' has no content", scenario_name, file_path)
        #         continue
        #
        #     row_count = sum(1 for row in results)
        #     row_count = row_count - 1
        #     file.seek(0)
        #     results = csv.reader(file, delimiter=',')
        #     first_row = next(results)
        #     initial_ts =  first_row[0]
        #     # rtt_per_sec = {} #NEW
        #     for interval in islice(results, 0, row_count, None):
        #         curr_ts = interval[0]
        #         duration = float(curr_ts) - float(initial_ts)
        #         rtt = float(interval[1])
        #         # rtt_per_sec[duration] = max(rtt_per_sec.get(duration, 0), rtt) #NEW
        #         df = df.append({
        #             'run': run,
        #             'second': duration,
        #             'avg_rtt': rtt,
        #         }, ignore_index=True)

        # Remove intermediate files
        files = os.listdir(path)
        for file in files:
            if file.endswith(".csv"):
                os.remove(os.path.join(path, file))

        # for key in rtt_per_sec:
        #     df = df.append({
        #             'run': run,
        #             'second': key,
        #             'avg_rtt': rtt_per_sec[key],
        #        }, ignore_index=True)

    if df.empty:
        logger.warning("%s: No MPTCP%s DL RTT data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_rtt_ul(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                       multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP UL RTT values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP UL rtt results")

    df_cols = [*config_cols, 'run', 'second', 'avg_rtt']
    if multi_process:
        df_mptcp_rtt_ul = __mp_parse_slices(4, __parse_mptcp_rtt_ul_from_pcaps, in_dir, compression, run_type, scenarios,
                                            df_cols, 'mptcp-ul', 'rtt_ul')
    else:
        df_mptcp_rtt_ul = __parse_slice(__parse_mptcp_rtt_ul_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                        df_cols, 'mptcp-ul', 'rtt_ul')

    logger.debug("Fixing MPTCP UL rtt data types")
    df_mptcp_rtt_ul = fix_dtypes(df_mptcp_rtt_ul)

    logger.info("Saving MPTCP UL rtt data")
    df_mptcp_rtt_ul.to_pickle(os.path.join(out_dir, 'mptcp_rtt_ul.pkl'))
    with open(os.path.join(out_dir, 'mptcp_rtt_ul.csv'), 'w+') as out_file:
        df_mptcp_rtt_ul.to_csv(out_file)

    return df_mptcp_rtt_ul


def __parse_mptcp_rtt_ul_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP UL RTT results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s UL RTT files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_rtt'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue
        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_rtt_seq_ts_0.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -r 2 -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -r 2 -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        df_i = pd.read_csv(processed_file_name, header=None)
        df_i.iloc[:, 0] = df_i.iloc[:, 0] - df_i.iloc[0, 0]

        df['second'] = df_i.iloc[:, 0]
        df['avg_rtt'] = df_i.iloc[:, 1]
        df = df.assign(run=run)

        # with open(processed_file_name, 'r') as file:
        #     results = csv.reader(file, delimiter=',')
        #     if results is None:
        #         logger.warning("%s: '%s' has no content", scenario_name, file_path)
        #         continue
        #
        #     row_count = sum(1 for row in results)
        #     row_count = row_count - 1
        #     file.seek(0)
        #     results = csv.reader(file, delimiter=',')
        #     first_row = next(results)
        #     initial_ts =  first_row[0]
        #     # rtt_per_sec = {} #NEW
        #    for interval in islice(results, 0, row_count, 100):
        #         curr_ts = interval[0]
        #         duration = float(curr_ts) - float(initial_ts)
        #         duration = int(duration)
        #         rtt = float(interval[1])
        #         # rtt_per_sec[duration] = max(rtt_per_sec.get(duration, 0), rtt) #NEW
        #         df = df.append({
        #             'run': run,
        #             'second': duration,
        #             'avg_rtt': rtt,
        #         }, ignore_index=True)

        # Remove intermediate files
        files = os.listdir(path)
        for file in files:
            if file.endswith(".csv"):
                os.remove(os.path.join(path, file))

        # for key in rtt_per_sec:
        #     df = df.append({
        #             'run': run,
        #             'second': key,
        #             'avg_rtt': rtt_per_sec[key],
        #         }, ignore_index=True)

    if df.empty:
        logger.warning("%s: No MPTCP%s UL RTT data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_owd_dl(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                       multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP DL OWD values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP DL OWD results")

    df_cols = [*config_cols, 'run', 'second', 'avg_owd']
    if multi_process:
        df_mptcp_owd_dl = __mp_parse_slices(4, __parse_mptcp_owd_dl_from_pcaps, in_dir, compression, run_type, scenarios,
                                            df_cols, 'mptcp-dl', 'owd_dl')
    else:
        df_mptcp_owd_dl = __parse_slice(__parse_mptcp_owd_dl_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                        df_cols, 'mptcp-dl', 'owd_dl')

    logger.debug("Fixing MPTCP DL OWD data types")
    df_mptcp_owd_dl = fix_dtypes(df_mptcp_owd_dl)

    logger.info("Saving MPTCP DL OWD data")
    df_mptcp_owd_dl.to_pickle(os.path.join(out_dir, 'mptcp_owd_dl.pkl'))
    with open(os.path.join(out_dir, 'mptcp_owd_dl.csv'), 'w+') as out_file:
        df_mptcp_owd_dl.to_csv(out_file)

    return df_mptcp_owd_dl


def __parse_mptcp_owd_dl_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP DL OWD results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse MPTCP or MPTCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s DL OWD files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_owd'])

    seq_send_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_seq_0.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = interval[1]
                if not seq_send_ts[seq_no]:
                    seq_send_ts[seq_no].append(curr_ts)

            # Remove intermediate files
            files = os.listdir(path)
            for file in files:
                if file.endswith(".csv"):
                    os.remove(os.path.join(path, file))

    seq_rcv_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_seq_1.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = interval[1]
                ack_no = float(interval[4])
                if not seq_rcv_ts[seq_no] and ack_no != 0:
                    seq_rcv_ts[seq_no].append(curr_ts)

            # Remove intermediate files
            files = os.listdir(path)
            for file in files:
                if file.endswith(".csv"):
                    os.remove(os.path.join(path, file))

    # for key in seq_send_ts:
    #     try:
    #         send_ts = seq_send_ts[key][0]
    #         rcv_ts = seq_rcv_ts[key][0]
    #         owd = float(rcv_ts) - float(send_ts)
    #         df = df.append({
    #                 'run': run,
    #                'second': float(send_ts),
    #                 'avg_owd': owd,
    #             }, ignore_index=True)
    #     except:
    #         continue

    try:
        df_seq_snd_ts = pd.DataFrame.from_dict(seq_send_ts, orient="index")
        df_seq_rcv_ts = pd.DataFrame.from_dict(seq_rcv_ts, orient="index")
        df_tmp = df_seq_snd_ts.join(df_seq_rcv_ts, lsuffix='_left')
        df_tmp.iloc[:, 1] = df_tmp.iloc[:, 1].astype(
            float) - df_tmp.iloc[:, 0].astype(float)
        df_tmp.iloc[:, 0] = df_tmp.iloc[:, 0].astype(
            float) - float(df_tmp.iloc[0, 0])

        df['second'] = df_tmp.iloc[:, 0]
        df['avg_owd'] = df_tmp.iloc[:, 1] * 1000
        df = df.assign(run=run)
    except:
        pass

    if df.empty:
        logger.warning("%s: No MPTCP%s DL OWD data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_owd_ul(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                       multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP UL OWD values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP UL owd results")

    df_cols = [*config_cols, 'run', 'second', 'avg_owd']
    if multi_process:
        df_mptcp_owd_ul = __mp_parse_slices(4, __parse_mptcp_owd_ul_from_pcaps, in_dir, compression, run_type, scenarios,
                                            df_cols, 'mptcp-ul', 'owd_ul')
    else:
        df_mptcp_owd_ul = __parse_slice(__parse_mptcp_owd_ul_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                        df_cols, 'mptcp-ul', 'owd_ul')

    logger.debug("Fixing MPTCP UL OWD data types")
    df_mptcp_owd_ul = fix_dtypes(df_mptcp_owd_ul)

    logger.info("Saving MPTCP UL OWD data")
    df_mptcp_owd_ul.to_pickle(os.path.join(out_dir, 'mptcp_owd_ul.pkl'))
    with open(os.path.join(out_dir, 'mptcp_owd_ul.csv'), 'w+') as out_file:
        df_mptcp_owd_ul.to_csv(out_file)

    return df_mptcp_owd_ul


def __parse_mptcp_owd_ul_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP UL OWD results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s UL owd files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_owd'], dtype=object)

    seq_send_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_seq_0.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = interval[1]
                if not seq_send_ts[seq_no]:
                    seq_send_ts[seq_no].append(curr_ts)

            # Remove intermediate files
            files = os.listdir(path)
            for file in files:
                if file.endswith(".csv"):
                    os.remove(os.path.join(path, file))

    seq_rcv_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_seq_1.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = interval[1]
                ack_no = float(interval[4])
                if not seq_rcv_ts[seq_no] and ack_no != 0:
                    seq_rcv_ts[seq_no].append(curr_ts)

            # Remove intermediate files
            files = os.listdir(path)
            for file in files:
                if file.endswith(".csv"):
                    os.remove(os.path.join(path, file))

    try:
        df_seq_snd_ts = pd.DataFrame.from_dict(seq_send_ts, orient="index")
        df_seq_rcv_ts = pd.DataFrame.from_dict(seq_rcv_ts, orient="index")
        df_tmp = df_seq_snd_ts.join(df_seq_rcv_ts, lsuffix='_left')
        df_tmp.iloc[:, 1] = df_tmp.iloc[:, 1].astype(
            float) - df_tmp.iloc[:, 0].astype(float)
        df_tmp.iloc[:, 0] = df_tmp.iloc[:, 0].astype(
            float) - float(df_tmp.iloc[0, 0])

        df['second'] = df_tmp.iloc[:, 0]
        df['avg_owd'] = df_tmp.iloc[:, 1] * 1000
        df = df.assign(run=run)
    except:
        pass

    if df.empty:
        logger.warning("%s: No MPTCP%s UL owd data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_ofo_queue_ul(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                             multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP UL OFO queue values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP UL ofo queue results")

    df_cols = [*config_cols, 'run', 'second', 'ofo_queue_size']
    if multi_process:
        df_mptcp_ofo_ul = __mp_parse_slices(4, __parse_mptcp_ofo_ul_from_pcaps, in_dir, compression, run_type, scenarios,
                                            df_cols, 'mptcp-ul', 'ofo_ul')
    else:
        df_mptcp_ofo_ul = __parse_slice(__parse_mptcp_ofo_ul_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                        df_cols, 'mptcp-ul', 'ofo_ul')

    logger.debug("Fixing MPTCP UL OFO queue types")
    df_mptcp_ofo_ul = fix_dtypes(df_mptcp_ofo_ul)

    logger.info("Saving MPTCP UL OFO queue data")
    df_mptcp_ofo_ul.to_pickle(os.path.join(out_dir, 'mptcp_ofo_ul.pkl'))
    with open(os.path.join(out_dir, 'mptcp_ofo_ul.csv'), 'w+') as out_file:
        df_mptcp_ofo_ul.to_csv(out_file)

    return df_mptcp_ofo_ul


def __parse_mptcp_ofo_ul_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP UL OFO results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s UL ofo files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(
        columns=['run', 'second', 'ofo_queue_size'], dtype=object)

    seq_send = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_seq_1.csv"

        # Decompress PCAP file using xz
        if int(compression) == 1:
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = interval[1]
                if not seq_send[seq_no]:
                    seq_send[seq_no].append(curr_ts)

            # Remove intermediate files
            # files = os.listdir(path)
            # for file in files:
            #     if file.endswith(".csv"):
            #         os.remove(os.path.join(path, file))

    seq_ofo = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_mptcp_queue_occ\.log.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_mptcp_queue_occ\.log$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # MPTPCP OFO file name
        processed_file_name = path + "/" + file_name

        # Decompress PCAP file using xz
        if int(compression) == 1:
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)
            file_path = file_path[:-3]

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=' ')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=' ')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                ts = interval[0]
                queue_probe_id = interval[1]
                seq_no = interval[2]
                ack_no = interval[3]
                queue_size = interval[4]
                if seq_no not in seq_ofo:
                    if queue_probe_id == '1':
                        seq_ofo[seq_no].append(queue_size)

        # Compress PCAP file using xz
        if int(compression) == 1:
            compress_cmd = "xz -T0 " + file_path
            os.system(compress_cmd)

            # Remove intermediate files
            # files = os.listdir(path)
            # for file in files:
            #     if file.endswith(".csv"):
            #         os.remove(os.path.join(path, file))

    try:
        df_seq_snd = pd.DataFrame.from_dict(seq_send, orient="index")
        print(df_seq_snd)
        df_seq_ofo = pd.DataFrame.from_dict(seq_ofo, orient="index")
        print(df_seq_ofo)
        # df_tmp = df_seq_snd.join(df_seq_ofo, how='inner')
        df_tmp = df_seq_snd.join(df_seq_ofo, how='inner', lsuffix='_right')
        print(df_tmp)
        df_tmp.iloc[:, 0] = df_tmp.iloc[:, 0].astype(
            float) - float(df_tmp.iloc[0, 0])
        print(df_tmp)

        df['second'] = df_tmp.iloc[:, 0]
        df['ofo_queue_size'] = df_tmp.iloc[:, 1]
        df = df.assign(run=run)
    except:
        pass

    if df.empty:
        logger.warning("%s: No MPTCP%s UL ofo queue data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_ofo_queue_dl(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                             multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP DL OFO queue values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP DL ofo queue results")

    df_cols = [*config_cols, 'run', 'second', 'ofo_queue_size']
    if multi_process:
        df_mptcp_ofo_dl = __mp_parse_slices(4, __parse_mptcp_ofo_dl_from_pcaps, in_dir, compression, run_type, scenarios,
                                            df_cols, 'mptcp-dl', 'ofo_dl')
    else:
        df_mptcp_ofo_dl = __parse_slice(__parse_mptcp_ofo_dl_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                        df_cols, 'mptcp-dl', 'ofo_dl')

    logger.debug("Fixing MPTCP DL OFO queue types")
    df_mptcp_ofo_dl = fix_dtypes(df_mptcp_ofo_dl)

    logger.info("Saving MPTCP DL OFO queue data")
    df_mptcp_ofo_dl.to_pickle(os.path.join(out_dir, 'mptcp_ofo_dl.pkl'))
    with open(os.path.join(out_dir, 'mptcp_ofo_dl.csv'), 'w+') as out_file:
        df_mptcp_ofo_dl.to_csv(out_file)

    return df_mptcp_ofo_dl


def __parse_mptcp_ofo_dl_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP DL OFO results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s DL ofo files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(
        columns=['run', 'second', 'ofo_queue_size'], dtype=object)

    seq_send = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file name
        processed_file_name = path + "/c2s_seq_0.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = interval[1]
                if not seq_send[seq_no]:
                    seq_send[seq_no].append(curr_ts)

            # Remove intermediate files
            files = os.listdir(path)
            for file in files:
                if file.endswith(".csv"):
                    os.remove(os.path.join(path, file))

    seq_ofo = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_mptcp_queue_occ\.log.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_mptcp_queue_occ\.log$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # MPTPCP OFO file name
        processed_file_name = path + "/" + file_name

        # Decompress file using xz
        if int(compression) == 1:
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)
            file_path = file_path[:-3]

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=' ')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=' ')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                ts = interval[0]
                queue_probe_id = interval[1]
                seq_no = interval[2]
                ack_no = interval[3]
                queue_size = interval[4]
                if seq_no not in seq_ofo:
                    if queue_probe_id == '1':
                        seq_ofo[seq_no].append(queue_size)

            # Remove intermediate files
            # files = os.listdir(path)
            # for file in files:
            #     if file.endswith(".csv"):
            #         os.remove(os.path.join(path, file))

        # Compress file using xz
        if int(compression) == 1:
            compress_cmd = "xz -T0 " + file_path
            os.system(compress_cmd)

    try:
        df_seq_snd = pd.DataFrame.from_dict(seq_send, orient="index")
        print(df_seq_snd)
        df_seq_ofo = pd.DataFrame.from_dict(seq_ofo, orient="index")
        print(df_seq_ofo)
        # df_tmp = df_seq_snd.join(df_seq_ofo, how='inner', lsuffix='_left)
        df_tmp = df_seq_snd.join(df_seq_ofo, how='inner', lsuffix='_right')
        print(df_tmp)
        df_tmp.iloc[:, 0] = df_tmp.iloc[:, 0].astype(
            float) - float(df_tmp.iloc[0, 0])
        print(df_tmp)

        df['second'] = df_tmp.iloc[:, 0]
        df['ofo_queue_size'] = df_tmp.iloc[:, 1]
        df = df.assign(run=run)
    except:
        pass

    if df.empty:
        logger.warning("%s: No MPTCP%s DL ofo queue data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_owd_dl_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                         multi_process: bool = False) -> pd.DataFrame:
    """
    Parse TCP DL LTE OWD values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing TCP DL LTE OWD owd results")

    df_cols = [*config_cols, 'run', 'second', 'avg_owd']
    if multi_process:
        df_tcp_owd_dl_lte = __mp_parse_slices(4, __parse_tcp_owd_dl_lte_from_pcaps, in_dir, compression, run_type, scenarios,
                                              df_cols, 'tcp-dl-lte', 'owd_dl_lte')
    else:
        df_tcp_owd_dl_lte = __parse_slice(__parse_tcp_owd_dl_lte_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                          df_cols, 'tcp-dl-lte', 'owd_dl_lte')

    logger.debug("Fixing TCP DL LTE OWD data types")
    df_tcp_owd_dl_lte = fix_dtypes(df_tcp_owd_dl_lte)

    logger.info("Saving TCP DL LTE OWD data")
    df_tcp_owd_dl_lte.to_pickle(os.path.join(out_dir, 'tcp_owd_dl_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_owd_dl_lte.csv'), 'w+') as out_file:
        df_tcp_owd_dl_lte.to_csv(out_file)

    return df_tcp_owd_dl_lte


def __parse_tcp_owd_dl_lte_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the TCP DL LTE OWD results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing TCP%s DL LTE OWD files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_owd'], dtype=object)

    seq_send_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_sender_owd_dl_lte.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.dst==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.IPERF_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_sender_owd_dl_lte.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.dst==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.IPERF_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = int(interval[1])
                tcp_len = int(interval[3])
                seq_no = seq_no + tcp_len
                if not seq_send_ts[seq_no]:
                    seq_send_ts[seq_no].append(curr_ts)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    seq_rcv_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)

        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_receiver_owd_dl_lte.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + common.SERVER_IP + \
                " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + " > " + \
                processed_file_name if int(run_type) == 4 \
                else "tshark -r " + file_path[:-3] + " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + common.SERVER_IP_MP + \
                " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + " > " + \
                processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_receiver_owd_dl_lte.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + common.SERVER_IP + \
                " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + " > " + \
                processed_file_name if int(run_type) == 4 \
                else "tshark -r " + file_path + " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + common.SERVER_IP_MP + \
                " and ip.dst==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + " > " + \
                processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = int(interval[1])
                ack_no = int(interval[2])
                if not seq_rcv_ts[seq_no]:
                    seq_rcv_ts[seq_no].append(curr_ts)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    try:
        df_seq_snd_ts = pd.DataFrame.from_dict(seq_send_ts, orient="index")
        df_seq_rcv_ts = pd.DataFrame.from_dict(seq_rcv_ts, orient="index")
        df_tmp = df_seq_snd_ts.join(df_seq_rcv_ts, lsuffix='_left')
        df_tmp.iloc[:, 1] = df_tmp.iloc[:, 1].astype(
            float) - df_tmp.iloc[:, 0].astype(float)
        df_tmp.iloc[:, 0] = df_tmp.iloc[:, 0].astype(
            float) - float(df_tmp.iloc[0, 0])

        df['second'] = df_tmp.iloc[:, 0]
        df['avg_owd'] = df_tmp.iloc[:, 1] * 1000
        df = df.assign(run=run)
    except:
        pass

    if df.empty:
        logger.warning("%s: No TCP%s DL LTE owd data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_owd_dl_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                         multi_process: bool = False) -> pd.DataFrame:
    """
    Parse TCP DL SAT OWD values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing TCP DL SAT OWD owd results")

    df_cols = [*config_cols, 'run', 'second', 'avg_owd']
    if multi_process:
        df_tcp_owd_dl_sat = __mp_parse_slices(4, __parse_tcp_owd_dl_sat_from_pcaps, in_dir, compression, run_type, scenarios,
                                              df_cols, 'tcp-dl-sat', 'owd_dl_sat')
    else:
        df_tcp_owd_dl_sat = __parse_slice(__parse_tcp_owd_dl_sat_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                          df_cols, 'tcp-dl-sat', 'owd_dl_sat')

    logger.debug("Fixing TCP DL SAT OWD data types")
    df_tcp_owd_dl_sat = fix_dtypes(df_tcp_owd_dl_sat)

    logger.info("Saving TCP DL SAT OWD data")
    df_tcp_owd_dl_sat.to_pickle(os.path.join(out_dir, 'tcp_owd_dl_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_owd_dl_sat.csv'), 'w+') as out_file:
        df_tcp_owd_dl_sat.to_csv(out_file)

    return df_tcp_owd_dl_sat


def __parse_tcp_owd_dl_sat_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the TCP DL SAT OWD results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing TCP%s DL SAT OWD files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_owd'], dtype=object)

    seq_send_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_sender_owd_dl_sat.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.dst==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.IPERF_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file names
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_sender_owd_dl_sat.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.dst==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.IPERF_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = int(interval[1])
                tcp_len = int(interval[3])
                seq_no = seq_no + tcp_len
                if not seq_send_ts[seq_no]:
                    seq_send_ts[seq_no].append(curr_ts)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    seq_rcv_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file names
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_receiver_owd_dl_sat.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + common.SERVER_IP + \
                " and ip.dst==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + " > " + \
                processed_file_name if int(run_type) == 3 \
                else "tshark -r " + file_path[:-3] + " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + common.SERVER_IP_MP + \
                " and ip.dst==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + " > " + \
                processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_receiver_owd_dl_sat.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + common.SERVER_IP + \
                " and ip.dst==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + " > " + \
                processed_file_name if int(run_type) == 3 \
                else "tshark -r " + file_path + " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + common.SERVER_IP_MP + \
                " and ip.dst==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + " > " + \
                processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = int(interval[1])
                ack_no = int(interval[2])
                if not seq_rcv_ts[seq_no]:
                    seq_rcv_ts[seq_no].append(curr_ts)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    try:
        df_seq_snd_ts = pd.DataFrame.from_dict(seq_send_ts, orient="index")
        df_seq_rcv_ts = pd.DataFrame.from_dict(seq_rcv_ts, orient="index")
        df_tmp = df_seq_snd_ts.join(df_seq_rcv_ts, lsuffix='_left')
        df_tmp.iloc[:, 1] = df_tmp.iloc[:, 1].astype(
            float) - df_tmp.iloc[:, 0].astype(float)
        df_tmp.iloc[:, 0] = df_tmp.iloc[:, 0].astype(
            float) - float(df_tmp.iloc[0, 0])

        df['second'] = df_tmp.iloc[:, 0]
        df['avg_owd'] = df_tmp.iloc[:, 1] * 1000
        df = df.assign(run=run)
    except:
        pass

    if df.empty:
        logger.warning("%s: No TCP%s DL SAT OWD data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_owd_ul_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                         multi_process: bool = False) -> pd.DataFrame:
    """
    Parse TCP UL LTE OWD values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing TCP UL LTE OWD owd results")

    df_cols = [*config_cols, 'run', 'second', 'avg_owd']
    if multi_process:
        df_tcp_owd_ul_lte = __mp_parse_slices(4, __parse_tcp_owd_ul_lte_from_pcaps, in_dir, compression, run_type, scenarios,
                                              df_cols, 'tcp-ul-lte', 'owd_ul_lte')
    else:
        df_tcp_owd_ul_lte = __parse_slice(__parse_tcp_owd_ul_lte_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                          df_cols, 'tcp-ul-lte', 'owd_ul_lte')

    logger.debug("Fixing TCP UL LTE OWD data types")
    df_tcp_owd_ul_lte = fix_dtypes(df_tcp_owd_ul_lte)

    logger.info("Saving TCP UL LTE OWD data")
    df_tcp_owd_ul_lte.to_pickle(os.path.join(out_dir, 'tcp_owd_ul_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_owd_ul_lte.csv'), 'w+') as out_file:
        df_tcp_owd_ul_lte.to_csv(out_file)

    return df_tcp_owd_ul_lte


def __parse_tcp_owd_ul_lte_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the TCP UL LTE OWD results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing TCP%s UL LTE OWD files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_owd'], dtype=object)

    seq_send_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)

        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_sender_owd_ul_lte.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.src==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_sender_owd_ul_lte.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.src==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = float(interval[1])
                tcp_len = float(interval[3])
                seq_no = seq_no + tcp_len
                if not seq_send_ts[seq_no]:
                    seq_send_ts[seq_no].append(curr_ts)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    seq_rcv_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_receiver_owd_ul_lte.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_receiver_owd_ul_lte.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = float(interval[1])
                ack_no = float(interval[2])
                if not seq_rcv_ts[seq_no]:
                    seq_rcv_ts[seq_no].append(curr_ts)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    try:
        df_seq_snd_ts = pd.DataFrame.from_dict(seq_send_ts, orient="index")
        df_seq_rcv_ts = pd.DataFrame.from_dict(seq_rcv_ts, orient="index")
        df_tmp = df_seq_snd_ts.join(df_seq_rcv_ts, lsuffix='_left')
        df_tmp.iloc[:, 1] = df_tmp.iloc[:, 1].astype(
            float) - df_tmp.iloc[:, 0].astype(float)
        df_tmp.iloc[:, 0] = df_tmp.iloc[:, 0].astype(
            float) - float(df_tmp.iloc[0, 0])

        df['second'] = df_tmp.iloc[:, 0]
        df['avg_owd'] = df_tmp.iloc[:, 1] * 1000
        df = df.assign(run=run)
    except:
        pass

    if df.empty:
        logger.warning("%s: No TCP%s UL LTE owd data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_owd_ul_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                         multi_process: bool = False) -> pd.DataFrame:
    """
    Parse TCP UL SAT OWD values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing TCP UL SAT OWD owd results")

    df_cols = [*config_cols, 'run', 'second', 'avg_owd']
    if multi_process:
        df_tcp_owd_ul_sat = __mp_parse_slices(4, __parse_tcp_owd_ul_sat_from_pcaps, in_dir, compression, run_type, scenarios,
                                              df_cols, 'tcp-ul-sat', 'owd_ul_sat')
    else:
        df_tcp_owd_ul_sat = __parse_slice(__parse_tcp_owd_ul_sat_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                          df_cols, 'tcp-ul-sat', 'owd_ul_sat')

    logger.debug("Fixing TCP UL SAT OWD data types")
    df_tcp_owd_ul_sat = fix_dtypes(df_tcp_owd_ul_sat)

    logger.info("Saving TCP UL SAT OWD data")
    df_tcp_owd_ul_sat.to_pickle(os.path.join(out_dir, 'tcp_owd_ul_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_owd_ul_sat.csv'), 'w+') as out_file:
        df_tcp_owd_ul_sat.to_csv(out_file)

    return df_tcp_owd_ul_sat


def __parse_tcp_owd_ul_sat_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the TCP UL SAT OWD results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing TCP%s UL SAT OWD files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_owd'], dtype=object)

    seq_send_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)

        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_sender_owd_ul_sat.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.src==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_sender_owd_ul_sat.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.src==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = float(interval[1])
                tcp_len = float(interval[3])
                seq_no = seq_no + tcp_len
                if not seq_send_ts[seq_no]:
                    seq_send_ts[seq_no].append(curr_ts)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    seq_rcv_ts = defaultdict(list)
    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_receiver_owd_ul_sat.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_receiver_owd_ul_sat.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack ip.src==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            first_row = next(results)
            for interval in islice(results, 0, row_count, None):
                curr_ts = interval[0]
                seq_no = float(interval[1])
                ack_no = float(interval[2])
                if not seq_rcv_ts[seq_no]:
                    seq_rcv_ts[seq_no].append(curr_ts)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    try:
        df_seq_snd_ts = pd.DataFrame.from_dict(seq_send_ts, orient="index")
        df_seq_rcv_ts = pd.DataFrame.from_dict(seq_rcv_ts, orient="index")
        df_tmp = df_seq_snd_ts.join(df_seq_rcv_ts, lsuffix='_left')
        df_tmp.iloc[:, 1] = df_tmp.iloc[:, 1].astype(
            float) - df_tmp.iloc[:, 0].astype(float)
        df_tmp.iloc[:, 0] = df_tmp.iloc[:, 0].astype(
            float) - float(df_tmp.iloc[0, 0])

        df['second'] = df_tmp.iloc[:, 0]
        df['avg_owd'] = df_tmp.iloc[:, 1] * 1000
        df = df.assign(run=run)
    except:
        pass

    if df.empty:
        logger.warning("%s: No TCP%s UL SAT OWD data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_rtt_ul_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                         multi_process: bool = False) -> pd.DataFrame:
    """
    Parse PCAP at the server to process RTTs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract RTT UL values on LTE flow")

    df_cols = [*config_cols, 'run', 'second', 'avg_rtt']
    if multi_process:
        df_rtt_ul_lte = __mp_parse_slices(4, __parse_tcp_rtt_ul_lte_from_pcaps, in_dir, compression, run_type, scenarios,
                                          df_cols, 'tcp-ul-lte', 'tcp_rtt_ul_lte')
    else:
        df_rtt_ul_lte = __parse_slice(__parse_tcp_rtt_ul_lte_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                      df_cols, 'tcp-ul-lte', 'tcp_rtt_ul_lte')

    logger.debug("Fixing tcp ul lte rtt data types")
    df_rtt_ul_lte = fix_dtypes(df_rtt_ul_lte)

    logger.info("Saving tcp ul lte rtt data")
    df_rtt_ul_lte.to_pickle(os.path.join(out_dir, 'tcp_rtt_ul_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_rtt_ul_lte.csv'), 'w+') as out_file:
        df_rtt_ul_lte.to_csv(out_file)

    return df_rtt_ul_lte


def __parse_tcp_rtt_ul_lte_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp lte results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s ul lte files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_rtt'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-8] + "_lte.log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_LTE + \
                " and tcp.port==" + common.GSTREAMER_PORT + "\",\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_SAT + \
                " and tcp.port==" + common.GSTREAMER_PORT + "\",\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and tcp.port==" + common.GSTREAMER_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-5] + "_lte.log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_LTE + \
                " and tcp.port==" + common.GSTREAMER_PORT + "\",\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_SAT + \
                " and tcp.port==" + common.GSTREAMER_PORT + "\",\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and tcp.port==" + common.GSTREAMER_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter='|')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 15, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                rtt = float(interval[2]) * bps_factor('K')
                df = df.append({
                    'run': run,
                    'second': duration,
                    'avg_rtt': rtt,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s ul lte rtt data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_rtt_ul_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                         multi_process: bool = False) -> pd.DataFrame:
    """
    Parse PCAP at the server to process RTTs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract RTT UL values on SAT flow")

    df_cols = [*config_cols, 'run', 'second', 'avg_rtt']
    if multi_process:
        df_rtt_ul_sat = __mp_parse_slices(4, __parse_tcp_rtt_ul_sat_from_pcaps, in_dir, compression, run_type, scenarios,
                                          df_cols, 'tcp-ul-sat', 'tcp_rtt_ul_sat')
    else:
        df_rtt_ul_sat = __parse_slice(__parse_tcp_rtt_ul_sat_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                      df_cols, 'tcp-ul-sat', 'tcp_rtt_ul_sat')

    logger.debug("Fixing tcp ul sat rtt data types")
    df_rtt_ul_sat = fix_dtypes(df_rtt_ul_sat)

    logger.info("Saving tcp ul sat rtt data")
    df_rtt_ul_sat.to_pickle(os.path.join(out_dir, 'tcp_rtt_ul_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_rtt_ul_sat.csv'), 'w+') as out_file:
        df_rtt_ul_sat.to_csv(out_file)

    return df_rtt_ul_sat


def __parse_tcp_rtt_ul_sat_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp ul sat results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s ul sat files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_rtt'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)

        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-8] + "_sat.log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_LTE + \
                " and tcp.port==" + common.GSTREAMER_PORT + "\",\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_SAT + \
                " and tcp.port==" + common.GSTREAMER_PORT + "\",\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and tcp.port==" + common.GSTREAMER_PORT + \
                "\" > " + processed_file_name
            # tshark -r tcp_duplex_1_dump_client_st3.pcap -T fields -E separator=, -e "tcp.analysis.ack_rtt" > rtt.log
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-5] + "_sat.log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_LTE + \
                " and tcp.port==" + common.GSTREAMER_PORT + "\",\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_SAT + \
                " and tcp.port==" + common.GSTREAMER_PORT + "\",\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and tcp.port==" + common.GSTREAMER_PORT + \
                "\" > " + processed_file_name
            # tshark -r tcp_duplex_1_dump_client_st3.pcap -T fields -E separator=, -e "tcp.analysis.ack_rtt" > rtt.log
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter='|')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 15, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                rtt = float(interval[3]) * bps_factor('K')
                df = df.append({
                    'run': run,
                    'second': duration,
                    'avg_rtt': rtt,
                }, ignore_index=True)

        # Remove intermediate files
            remove_cmd = "rm " + processed_file_name
            os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s ul sat rtt data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_rtt_dl_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                         multi_process: bool = False) -> pd.DataFrame:
    """
    Parse PCAP at the server to process RTTs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract RTT DL values on LTE flow")

    df_cols = [*config_cols, 'run', 'second', 'avg_rtt']
    if multi_process:
        df_rtt_dl_lte = __mp_parse_slices(4, __parse_tcp_rtt_dl_lte_from_pcaps, in_dir, compression, run_type, scenarios,
                                          df_cols, 'tcp-dl-lte', 'tcp_rtt_dl_lte')
    else:
        df_rtt_dl_lte = __parse_slice(__parse_tcp_rtt_dl_lte_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                      df_cols, 'tcp-dl-lte', 'tcp_rtt_dl_lte')

    logger.debug("Fixing tcp dl lte rtt data types")
    df_rtt_dl_lte = fix_dtypes(df_rtt_dl_lte)

    logger.info("Saving tcp dl rtt data")
    df_rtt_dl_lte.to_pickle(os.path.join(out_dir, 'tcp_rtt_dl_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_rtt_dl_lte.csv'), 'w+') as out_file:
        df_rtt_dl_lte.to_csv(out_file)

    return df_rtt_dl_lte


def __parse_tcp_rtt_dl_lte_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp lte results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s dl lte files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_rtt'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-8] + "_lte.log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -q -z io,stat,1,\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_LTE + \
                " and tcp.port==" + common.IPERF_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-5] + "_lte.log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -q -z io,stat,1,\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_LTE + \
                " and tcp.port==" + common.IPERF_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter='|')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 12, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                rtt = float(interval[2]) * bps_factor('K')
                df = df.append({
                    'run': run,
                    'second': duration,
                    'avg_rtt': rtt,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s lte rtt data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_rtt_dl_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                         multi_process: bool = False) -> pd.DataFrame:
    """
    Parse PCAP at the server to process RTTs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract RTT DL values on SAT flow")

    df_cols = [*config_cols, 'run', 'second', 'avg_rtt']
    if multi_process:
        df_rtt_dl_sat = __mp_parse_slices(4, __parse_tcp_rtt_dl_sat_from_pcaps, in_dir, compression, run_type, scenarios,
                                          df_cols, 'tcp-dl-sat', 'tcp_rtt_dl_sat')
    else:
        df_rtt_dl_sat = __parse_slice(__parse_tcp_rtt_dl_sat_from_pcaps, in_dir, compression, run_type, [*scenarios.items()],
                                      df_cols, 'tcp-dl-sat', 'tcp_rtt_dl_sat')

    logger.debug("Fixing tcp dl sat rtt data types")
    df_rtt_dl_sat = fix_dtypes(df_rtt_dl_sat)

    logger.info("Saving tcp dl rtt data")
    df_rtt_dl_sat.to_pickle(os.path.join(out_dir, 'tcp_rtt_dl_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_rtt_dl_sat.csv'), 'w+') as out_file:
        df_rtt_dl_sat.to_csv(out_file)

    return df_rtt_dl_sat


def __parse_tcp_rtt_dl_sat_from_pcaps(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp sat results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s dl sat files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'avg_rtt'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-8] + "_sat.log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -q -z io,stat,1,\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_SAT + \
                " and tcp.port==" + common.IPERF_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-5] + "_sat.log"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -q -z io,stat,1,\"MAX(tcp.analysis.ack_rtt) tcp.analysis.ack_rtt and ip.addr==" + common.CLIENT_IP_SAT + \
                " and tcp.port==" + common.IPERF_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter='|')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 13, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                rtt = float(interval[2]) * bps_factor('K')
                df = df.append({
                    'run': run,
                    'second': duration,
                    'avg_rtt': rtt,
                }, ignore_index=True)

        # Remove intermediate files
            remove_cmd = "rm " + processed_file_name
            os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s dl sat rtt data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_loss_dl(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                        multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP DL loss values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP DL loss results")

    df_cols = [*config_cols, 'run', 'second', 'retransmit', 'retransmit_rate',
               'r_3', 'r_3_rate', 'r_4', 'r_4_rate', 'r_300', 'r_300_rate']
    if multi_process:
        df_mptcp_loss_dl = __mp_parse_slices(4, __parse_mptcp_loss_dl_from_scenario, in_dir, compression, run_type, scenarios,
                                             df_cols, 'mptcp-dl', 'loss_dl')
    else:
        df_mptcp_loss_dl = __parse_slice(__parse_mptcp_loss_dl_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                         df_cols, 'mptcp-dl', 'loss_dl')

    logger.debug("Fixing MPTCP DL loss data types")
    df_mptcp_loss_dl = fix_dtypes(df_mptcp_loss_dl)

    logger.info("Saving MPTCP DL loss data")
    df_mptcp_loss_dl.to_pickle(os.path.join(out_dir, 'mptcp_loss_dl.pkl'))
    with open(os.path.join(out_dir, 'mptcp_loss_dl.csv'), 'w+') as out_file:
        df_mptcp_loss_dl.to_csv(out_file)

    return df_mptcp_loss_dl


def __parse_mptcp_loss_dl_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP DL loss results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s DL loss files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'retransmit',
                      'retransmit_rate', 'r_3', 'r_3_rate', 'r_4', 'r_4_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file path
        processed_file_name = path + "/c2s_seq_0.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            retransmits_per_sec = {}
            pkt_per_sec = {}
            first_row = next(results)
            initial_ts = first_row[0]
            retransmits_per_rto = {}
            tcp_rto = 0.2
            tcp_rto_300 = 0.3
            tcp_rto_3 = 1.4
            tcp_rto_4 = 3.0
            r_3 = {}
            r_4 = {}
            r_300 = {}

            for interval in islice(results, 0, row_count, None):
                try:
                    curr_ts = interval[0]
                    seq_no = interval[1]
                    is_retransmit = interval[3]
                    duration = float(curr_ts) - float(initial_ts)
                    duration = int(duration)
                    pkt_per_sec[duration] = pkt_per_sec.get(duration, 0) + 1

                    retransmits_per_sec[duration] = retransmits_per_sec.get(
                        duration, 0)
                    r_3[duration] = r_3.get(duration, 0)
                    r_4[duration] = r_4.get(duration, 0)
                    r_300[duration] = r_300.get(duration, 0)

                    # Check if packet is a retransmission by retransmit flag and if the
                    # sequence number repeats after a TCP RTO of 200 ms.
                    if (is_retransmit == '0'):
                        if seq_no not in retransmits_per_rto:
                            retransmits_per_rto[seq_no] = curr_ts
                            retransmits_per_sec[duration] = retransmits_per_sec.get(
                                duration, 0) + 1
                        else:
                            prev_ts = retransmits_per_rto.get(seq_no)
                            retransmit_interval = float(
                                curr_ts) - float(prev_ts)
                            if (retransmit_interval > tcp_rto):
                                retransmits_per_sec[duration] = retransmits_per_sec.get(
                                    duration, 0) + 1
                            if (retransmit_interval > tcp_rto_3):
                                r_3[duration] = r_3.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_4):
                                r_4[duration] = r_4.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_300):
                                r_300[duration] = r_300.get(duration, 0) + 1
                except IndexError:
                    pass

            # print(retransmits_per_sec)
            # print(r_3)
            # print(r_4)
            # logger.debug(retransmits_per_sec)
            # logger.debug(pkt_per_sec)
            for key in retransmits_per_sec:
                df = df.append({
                    'run': run,
                    'second': key,
                    'retransmit': retransmits_per_sec[key],
                    'retransmit_rate': float(retransmits_per_sec[key]/pkt_per_sec[key]),
                    # 'loss': pkt_losses_per_sec[key2],
                    # 'loss_rate': float(pkt_losses_per_sec[key2]/pkt_per_sec[key2]),
                    'r_3': r_3[key],
                    'r_3_rate': float(r_3[key]/pkt_per_sec[key]),
                    'r_4': r_4[key],
                    'r_4_rate': float(r_4[key]/pkt_per_sec[key]),
                    'r_300': r_300[key],
                    'r_300_rate': float(r_300[key]/pkt_per_sec[key]),
                }, ignore_index=True)
            # for key in pkt_losses_per_sec:
            #     df = df.append({
            #             'run': run,
            #             'second': key,
            #             'loss': float(pkt_losses_per_sec[key]/pkt_per_sec[key]),
            #         }, ignore_index=True)

            # val_retransmit_count = 0
            # for key in validate_retransmits:
            #     if (validate_retransmits[key] > 0):
            #         val_retransmit_count = val_retransmit_count + validate_retransmits[key]
            # logger.debug("%s: Validated retransmit count: %d", scenario_name, val_retransmit_count)

            # dup_ack_count = 0
            # for key in pkt_losses:
            #     if (pkt_losses[key] > 3):
            #         # print("key, value: ", key, pkt_losses[key])
            #        dup_ack_count = dup_ack_count + 1
            # logger.debug("%s: Packet loss count (signaled by >=3 duplicate ACKs): %d", scenario_name, dup_ack_count)

        # Remove intermediate files
        # remove_cmd = "rm " + processed_file_name
        # os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No MPTCP%s DL loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_mptcp_loss_ul(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                        multi_process: bool = False) -> pd.DataFrame:
    """
    Parse MPTCP UL loss values from PCAPs.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing MPTCP UL loss results")

    df_cols = [*config_cols, 'run', 'second', 'retransmit', 'retransmit_rate',
               'r_3', 'r_3_rate', 'r_4', 'r_4_rate', 'r_300', 'r_300_rate']
    if multi_process:
        df_mptcp_loss_ul = __mp_parse_slices(4, __parse_mptcp_loss_ul_from_scenario, in_dir, compression, run_type, scenarios,
                                             df_cols, 'mptcp-ul', 'loss_ul')
    else:
        df_mptcp_loss_ul = __parse_slice(__parse_mptcp_loss_ul_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                         df_cols, 'mptcp-ul', 'loss_ul')

    logger.debug("Fixing MPTCP UL loss data types")
    df_mptcp_loss_ul = fix_dtypes(df_mptcp_loss_ul)

    logger.info("Saving MPTCP UL loss data")
    df_mptcp_loss_ul.to_pickle(os.path.join(out_dir, 'mptcp_loss_ul.pkl'))
    with open(os.path.join(out_dir, 'mptcp_loss_ul.csv'), 'w+') as out_file:
        df_mptcp_loss_ul.to_csv(out_file)

    return df_mptcp_loss_ul


def __parse_mptcp_loss_ul_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the MPTCP UL loss results for the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing MPTCP%s UL loss files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'retransmit',
                      'retransmit_rate', 'r_3', 'r_3_rate', 'r_4', 'r_4_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name)
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # Mptcptrace intermediate file path
        processed_file_name = path + "/c2s_seq_1.csv"

        if int(compression) == 1:
            # Decompress PCAP file using xz
            decompress_cmd = "xz -T0 -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + \
                file_path[:-3] + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

            # Compress PCAP file using xz
            compress_cmd = "xz -T0 " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # Analyze PCAP using tshark
            extract_pcap_cmd = "mptcptrace -f " + file_path + " -s -w 2 > /dev/null 2>&1"
            p = subprocess.Popen([extract_pcap_cmd], cwd=path, shell=True)
            p.wait()

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            retransmits_per_sec = {}
            pkt_per_sec = {}
            first_row = next(results)
            initial_ts = first_row[0]
            retransmits_per_rto = {}
            tcp_rto = 0.2
            tcp_rto_300 = 0.3
            tcp_rto_3 = 1.4
            tcp_rto_4 = 3.0
            r_3 = {}
            r_4 = {}
            r_300 = {}

            for interval in islice(results, 0, row_count, None):
                try:
                    curr_ts = interval[0]
                    seq_no = interval[1]
                    is_retransmit = interval[3]
                    duration = float(curr_ts) - float(initial_ts)
                    duration = int(duration)
                    pkt_per_sec[duration] = pkt_per_sec.get(duration, 0) + 1

                    retransmits_per_sec[duration] = retransmits_per_sec.get(
                        duration, 0)
                    r_3[duration] = r_3.get(duration, 0)
                    r_4[duration] = r_4.get(duration, 0)
                    r_300[duration] = r_300.get(duration, 0)

                    # Check if packet is a retransmission by retransmit flag and if the
                    # sequence number repeats after a TCP RTO of 200 ms.
                    if (is_retransmit == '0'):
                        if seq_no not in retransmits_per_rto:
                            retransmits_per_rto[seq_no] = curr_ts
                            retransmits_per_sec[duration] = retransmits_per_sec.get(
                                duration, 0) + 1
                        else:
                            prev_ts = retransmits_per_rto.get(seq_no)
                            retransmit_interval = float(
                                curr_ts) - float(prev_ts)
                            if (retransmit_interval > tcp_rto):
                                retransmits_per_sec[duration] = retransmits_per_sec.get(
                                    duration, 0) + 1
                            if (retransmit_interval > tcp_rto_3):
                                r_3[duration] = r_3.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_4):
                                r_4[duration] = r_4.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_300):
                                r_300[duration] = r_300.get(duration, 0) + 1
                except IndexError:
                    pass

            # print(retransmits_per_sec)
            # print(r_3)
            # print(r_4)
            # logger.debug(retransmits_per_sec)
            # logger.debug(pkt_per_sec)
            for key in retransmits_per_sec:
                df = df.append({
                    'run': run,
                    'second': key,
                    'retransmit': retransmits_per_sec[key],
                    'retransmit_rate': float(retransmits_per_sec[key]/pkt_per_sec[key]),
                    # 'loss': pkt_losses_per_sec[key2],
                    # 'loss_rate': float(pkt_losses_per_sec[key2]/pkt_per_sec[key2]),
                    'r_3': r_3[key],
                    'r_3_rate': float(r_3[key]/pkt_per_sec[key]),
                    'r_4': r_4[key],
                    'r_4_rate': float(r_4[key]/pkt_per_sec[key]),
                    'r_300': r_300[key],
                    'r_300_rate': float(r_300[key]/pkt_per_sec[key]),
                }, ignore_index=True)
            # for key in pkt_losses_per_sec:
            #     df = df.append({
            #             'run': run,
            #             'second': key,
            #             'retransmits': float(pkt_losses_per_sec[key]/pkt_per_sec[key]),
            #         }, ignore_index=True)

            # val_retransmit_count = 0
            # for key in validate_retransmits:
            #     if (validate_retransmits[key] > 0):
            #         val_retransmit_count = val_retransmit_count + validate_retransmits[key]
            # logger.debug("%s: Validated retransmit count: %d", scenario_name, val_retransmit_count)

            # dup_ack_count = 0
            # for key in pkt_losses:
            #     if (pkt_losses[key] > 3):
            #         # print("key, value: ", key, pkt_losses[key])
            #        dup_ack_count = dup_ack_count + 1
            # logger.debug("%s: Packet loss count (signaled by >=3 duplicate ACKs): %d", scenario_name, dup_ack_count)

        # Remove intermediate files
        # remove_cmd = "rm " + processed_file_name
        # os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No MPTCP%s UL loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_losses_ul_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                            multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all PCAPs to extract packet losses.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract losses")

    # df_cols = [*config_cols, 'run', 'second', 'retransmits' ,'dup_acks', 'lost_segments', 'fast_retransmits', 'ofo_segments','estimated_losses']
    df_cols = [*config_cols, 'run', 'second', 'retransmission',
               'retransmission_rate', 'loss', 'loss_rate']

    if multi_process:
        df_tcp_loss_ul_sat = __mp_parse_slices(4, __parse_tcp_sat_losses_ul_from_scenario, in_dir, compression, run_type, scenarios,
                                               df_cols, 'tcp-ul-sat', 'tcp_loss_ul_sat')
    else:
        df_tcp_loss_ul_sat = __parse_slice(__parse_tcp_sat_losses_ul_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                           df_cols, 'tcp-ul-sat', 'tcp_loss_ul_sat')

    logger.debug("Fixing tcp ul satcom loss data types")
    df_tcp_loss_ul_sat = fix_dtypes(df_tcp_loss_ul_sat)

    logger.info("Saving tcp ul loss data")
    df_tcp_loss_ul_sat.to_pickle(os.path.join(out_dir, 'tcp_loss_ul_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_loss_ul_sat.csv'), 'w+') as out_file:
        df_tcp_loss_ul_sat.to_csv(out_file)

    return df_tcp_loss_ul_sat


def __parse_tcp_sat_losses_ul_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp ul satcom results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s ul satcom files in %s",
                 " (pep)" if pep else "", scenario_name)
    # df = pd.DataFrame(columns=['run', 'second', 'retransmits' ,'dup_acks', 'lost_segments', 'fast_retransmits', 'ofo_segments','estimated_losses'])
    df = pd.DataFrame(columns=[
                      'run', 'second', 'retransmission', 'retransmission_rate', 'loss', 'loss_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            # extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and tcp.port==4242\",\"COUNT(tcp.analysis.duplicate_ack)tcp.analysis.duplicate_ack and tcp.port==4242\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and tcp.port==4242\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and tcp.port==4242\",\"COUNT(tcp.analysis.out_of_order) tcp.analysis.out_of_order and tcp.port==4242\" > " + processed_file_name
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            # extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and tcp.port==4242\",\"COUNT(tcp.analysis.duplicate_ack)tcp.analysis.duplicate_ack and tcp.port==4242\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and tcp.port==4242\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and tcp.port==4242\",\"COUNT(tcp.analysis.out_of_order) tcp.analysis.out_of_order and tcp.port==4242\" > " + processed_file_name
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter='|')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 18, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                total_frames = int(interval[2])
                retransmissions = int(interval[4]) + int(interval[6])
                retransmission_rate = float(
                    retransmissions/total_frames) if total_frames else 0
                lost_segments = int(interval[5])
                loss_rate = float(
                    lost_segments/total_frames) if total_frames else 0
                df = df.append({
                    'run': run,
                    'second': duration,
                    'retransmission': retransmissions,
                    'retransmission_rate': retransmission_rate,
                    'loss': lost_segments,
                    'loss_rate': loss_rate,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s ul satcom loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_losses_ul_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                            multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all PCAPs to extract packet losses.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract losses")

    # df_cols = [*config_cols, 'run', 'second', 'retransmits' ,'dup_acks', 'lost_segments', 'fast_retransmits', 'ofo_segments','estimated_losses']
    df_cols = [*config_cols, 'run', 'second', 'retransmission',
               'retransmission_rate', 'loss', 'loss_rate']
    if multi_process:
        df_tcp_loss_ul_lte = __mp_parse_slices(4, __parse_tcp_lte_losses_ul_from_scenario, in_dir, compression, run_type, scenarios,
                                               df_cols, 'tcp-ul-lte', 'tcp_loss_ul_lte')
    else:
        df_tcp_loss_ul_lte = __parse_slice(__parse_tcp_lte_losses_ul_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                           df_cols, 'tcp-ul-lte', 'tcp_loss_ul_lte')

    logger.debug("Fixing tcp lte loss data types")
    df_tcp_loss_ul_lte = fix_dtypes(df_tcp_loss_ul_lte)

    logger.info("Saving tcp loss data")
    df_tcp_loss_ul_lte.to_pickle(os.path.join(out_dir, 'tcp_loss_ul_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_loss_ul_lte.csv'), 'w+') as out_file:
        df_tcp_loss_ul_lte.to_csv(out_file)

    return df_tcp_loss_ul_lte


def __parse_tcp_lte_losses_ul_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp lte results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s ul lte files in %s",
                 " (pep)" if pep else "", scenario_name)
    # df = pd.DataFrame(columns=['run', 'second', 'retransmits' ,'dup_acks', 'lost_segments', 'fast_retransmits', 'ofo_segments','estimated_losses'])
    df = pd.DataFrame(columns=[
                      'run', 'second', 'retransmission', 'retransmission_rate', 'loss', 'loss_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            # extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and tcp.port==4242\",\"COUNT(tcp.analysis.duplicate_ack)tcp.analysis.duplicate_ack and tcp.port==4242\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and tcp.port==4242\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and tcp.port==4242\",\"COUNT(tcp.analysis.out_of_order) tcp.analysis.out_of_order and tcp.port==4242\" > " + processed_file_name
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            # extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and tcp.port==4242\",\"COUNT(tcp.analysis.duplicate_ack)tcp.analysis.duplicate_ack and tcp.port==4242\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and tcp.port==4242\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and tcp.port==4242\",\"COUNT(tcp.analysis.out_of_order) tcp.analysis.out_of_order and tcp.port==4242\" > " + processed_file_name
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.GSTREAMER_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter='|')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 18, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                total_frames = int(interval[2])
                retransmissions = int(interval[4]) + int(interval[6])
                retransmission_rate = float(
                    retransmissions/total_frames) if total_frames else 0
                lost_segments = int(interval[5])
                loss_rate = float(
                    lost_segments/total_frames) if total_frames else 0
                df = df.append({
                    'run': run,
                    'second': duration,
                    'retransmission': retransmissions,
                    'retransmission_rate': retransmission_rate,
                    'loss': lost_segments,
                    'loss_rate': loss_rate,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s ul lte loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_losses_dl_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                            multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all PCAPs to extract packet losses.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract losses")

    # df_cols = [*config_cols, 'run', 'second', 'retransmits' ,'dup_acks', 'lost_segments', 'fast_retransmits', 'ofo_segments','estimated_losses']
    df_cols = [*config_cols, 'run', 'second', 'retransmission',
               'retransmission_rate', 'loss', 'loss_rate']

    if multi_process:
        df_tcp_loss_dl_sat = __mp_parse_slices(4, __parse_tcp_sat_losses_dl_from_scenario, in_dir, compression, run_type, scenarios,
                                               df_cols, 'tcp-dl-sat', 'tcp_loss_dl_sat')
    else:
        df_tcp_loss_dl_sat = __parse_slice(__parse_tcp_sat_losses_dl_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                           df_cols, 'tcp-dl-sat', 'tcp_loss_dl_sat')

    logger.debug("Fixing tcp dl satcom loss data types")
    df_tcp_loss_dl_sat = fix_dtypes(df_tcp_loss_dl_sat)

    logger.info("Saving tcp dl loss data")
    df_tcp_loss_dl_sat.to_pickle(os.path.join(out_dir, 'tcp_loss_dl_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_loss_dl_sat.csv'), 'w+') as out_file:
        df_tcp_loss_dl_sat.to_csv(out_file)

    return df_tcp_loss_dl_sat


def __parse_tcp_sat_losses_dl_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp dl satcom results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s dl satcom files in %s",
                 " (pep)" if pep else "", scenario_name)
    # df = pd.DataFrame(columns=['run', 'second', 'retransmits' ,'dup_acks', 'lost_segments', 'fast_retransmits', 'ofo_segments','estimated_losses'])
    df = pd.DataFrame(columns=[
                      'run', 'second', 'retransmission', 'retransmission_rate', 'loss', 'loss_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            # extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\",\"COUNT(tcp.analysis.duplicate_ack)tcp.analysis.duplicate_ack and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\",\"COUNT(tcp.analysis.out_of_order) tcp.analysis.out_of_order and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\" > " + processed_file_name
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            # extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\",\"COUNT(tcp.analysis.duplicate_ack)tcp.analysis.duplicate_ack and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\",\"COUNT(tcp.analysis.out_of_order) tcp.analysis.out_of_order and ip.src==" + common.CLIENT_IP_SAT + " and tcp.port==5201\" > " + processed_file_name
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.addr==" + common.CLIENT_IP_SAT + " and tcp.port==" + common.IPERF_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter='|')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 18, row_count, None):
                period = interval[1]
                duration = period[-5:].strip()
                if "Dur" in period:
                    continue
                total_frames = int(interval[2])
                retransmissions = int(interval[4]) + int(interval[6])
                retransmission_rate = float(
                    retransmissions/total_frames) if total_frames else 0
                lost_segments = int(interval[5])
                loss_rate = float(
                    lost_segments/total_frames) if total_frames else 0
                df = df.append({
                    'run': run,
                    'second': duration,
                    'retransmission': retransmissions,
                    'retransmission_rate': retransmission_rate,
                    'loss': lost_segments,
                    'loss_rate': loss_rate,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s dl satcom loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_losses_dl_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                            multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all PCAPs to extract packet losses.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract losses")

    # df_cols = [*config_cols, 'run', 'second', 'retransmits' ,'dup_acks', 'lost_segments', 'fast_retransmits', 'ofo_segments','estimated_losses']
    df_cols = [*config_cols, 'run', 'second', 'retransmission',
               'retransmission_rate', 'loss', 'loss_rate']

    if multi_process:
        df_tcp_loss_dl_lte = __mp_parse_slices(4, __parse_tcp_lte_losses_dl_from_scenario, in_dir, compression, run_type, scenarios,
                                               df_cols, 'tcp-dl-lte', 'tcp_loss_dl_lte')
    else:
        df_tcp_loss_dl_lte = __parse_slice(__parse_tcp_lte_losses_dl_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                           df_cols, 'tcp-dl-lte', 'tcp_loss_dl_lte')

    logger.debug("Fixing tcp lte loss data types")
    df_tcp_loss_dl_lte = fix_dtypes(df_tcp_loss_dl_lte)

    logger.info("Saving tcp loss data")
    df_tcp_loss_dl_lte.to_pickle(os.path.join(out_dir, 'tcp_loss_dl_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_loss_dl_lte.csv'), 'w+') as out_file:
        df_tcp_loss_dl_lte.to_csv(out_file)

    return df_tcp_loss_dl_lte


def __parse_tcp_lte_losses_dl_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp lte results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s dl lte files in %s",
                 " (pep)" if pep else "", scenario_name)
    # df = pd.DataFrame(columns=['run', 'second', 'retransmits' ,'dup_acks', 'lost_segments', 'fast_retransmits', 'ofo_segments','estimated_losses'])
    df = pd.DataFrame(columns=[
                      'run', 'second', 'retransmission', 'retransmission_rate', 'loss', 'loss_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-8] + ".log"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            # extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\",\"COUNT(tcp.analysis.duplicate_ack)tcp.analysis.duplicate_ack and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\",\"COUNT(tcp.analysis.out_of_order) tcp.analysis.out_of_order and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\" > " + processed_file_name
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + " -q -z io,stat,1,\"ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + \
                common.RAW_DATA_DIR + "/" + file_name[:-5] + ".log"

            # Analyze PCAP using tshark
            # extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\",\"COUNT(tcp.analysis.duplicate_ack)tcp.analysis.duplicate_ack and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\",\"COUNT(tcp.analysis.out_of_order) tcp.analysis.out_of_order and ip.src==" + common.CLIENT_IP_LTE + " and tcp.port==5201\" > " + processed_file_name
            extract_pcap_cmd = "tshark -r " + file_path + " -q -z io,stat,1,\"ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.lost_segment) tcp.analysis.lost_segment and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\",\"COUNT(tcp.analysis.fast_retransmission) tcp.analysis.fast_retransmission and ip.addr==" + common.CLIENT_IP_LTE + " and tcp.port==" + common.IPERF_PORT + \
                "\" > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter='|')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter='|')
            for interval in islice(results, 18, row_count, None):
                period = interval[1]
                duration = period[-3:].strip()
                if "Dur" in period:
                    continue
                total_frames = int(interval[2])
                retransmissions = int(interval[4]) + int(interval[6])
                retransmission_rate = float(
                    retransmissions/total_frames) if total_frames else 0
                lost_segments = int(interval[5])
                loss_rate = float(
                    lost_segments/total_frames) if total_frames else 0
                df = df.append({
                    'run': run,
                    'second': duration,
                    'retransmission': retransmissions,
                    'retransmission_rate': retransmission_rate,
                    'loss': lost_segments,
                    'loss_rate': loss_rate,
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s dl lte loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_losses_def_ul_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                                multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all PCAPs to extract packet losses.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract losses")

    df_cols = [*config_cols, 'run', 'second', 'retransmit', 'retransmit_rate',
               'r_3', 'r_3_rate', 'r_4', 'r_4_rate', 'r_300', 'r_300_rate']

    if multi_process:
        df_tcp_loss_def_ul_sat = __mp_parse_slices(4, __parse_tcp_sat_losses_def_ul_from_scenario, in_dir, compression, run_type, scenarios,
                                                   df_cols, 'tcp-ul-sat', 'tcp_loss_def_ul_sat')
    else:
        df_tcp_loss_def_ul_sat = __parse_slice(__parse_tcp_sat_losses_def_ul_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                               df_cols, 'tcp-ul-sat', 'tcp_loss_def_ul_sat')

    logger.debug("Fixing tcp ul satcom loss data types")
    df_tcp_loss_def_ul_sat = fix_dtypes(df_tcp_loss_def_ul_sat)

    logger.info("Saving tcp ul loss data")
    df_tcp_loss_def_ul_sat.to_pickle(
        os.path.join(out_dir, 'tcp_loss_def_ul_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_loss_def_ul_sat.csv'), 'w+') as out_file:
        df_tcp_loss_def_ul_sat.to_csv(out_file)

    return df_tcp_loss_def_ul_sat


def __parse_tcp_sat_losses_def_ul_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp ul satcom results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s ul satcom files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'retransmit',
                      'retransmit_rate', 'r_3', 'r_3_rate', 'r_4', 'r_4_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_st3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 3 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)

        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_sender_loss_ul_sat.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.src==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_sender_loss_ul_sat.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.src==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            retransmits_per_sec = dict.fromkeys(range(1800), 0)
            pkt_per_sec = dict.fromkeys(range(1800), 0)
            first_row = next(results)
            initial_ts = first_row[0]
            prev_seq_no = first_row[1]
            prev_seg_len = 1
            retransmits_per_rto = {}
            tcp_rto = 0.2
            tcp_rto_300 = 0.3
            tcp_rto_3 = 1.4
            tcp_rto_4 = 3.0
            r_3 = dict.fromkeys(range(1800), 0)
            r_4 = dict.fromkeys(range(1800), 0)
            r_300 = dict.fromkeys(range(1800), 0)
            for interval in islice(results, 0, row_count, None):
                try:
                    curr_ts = interval[0]
                    seq_no = float(interval[1])
                    seg_len = interval[3]
                    duration = float(curr_ts) - float(initial_ts)
                    duration = int(duration)
                    pkt_per_sec[duration] = pkt_per_sec.get(duration, 0) + 1

                    # Check if packet is a retransmission by retransmit flag and if the
                    # sequence number repeats after a TCP RTO of 200 ms.
                    expected_seq_no = float(prev_seq_no) + float(prev_seg_len)
                    # print(seq_no, expected_seq_no, prev_seq_no, prev_seg_len)
                    # print(type(seq_no), type(expected_seq_no))
                    # print(seq_no == expected_seq_no)
                    if seq_no != expected_seq_no:
                        if seq_no not in retransmits_per_rto:
                            retransmits_per_rto[seq_no] = curr_ts
                            retransmits_per_sec[duration] = retransmits_per_sec.get(
                                duration, 0) + 1
                        else:
                            prev_ts = retransmits_per_rto.get(seq_no)
                            retransmit_interval = float(
                                curr_ts) - float(prev_ts)
                            if (retransmit_interval > tcp_rto):
                                retransmits_per_sec[duration] = retransmits_per_sec.get(
                                    duration, 0) + 1
                            if (retransmit_interval > tcp_rto_3):
                                r_3[duration] = r_3.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_4):
                                r_4[duration] = r_4.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_300):
                                r_300[duration] = r_300.get(duration, 0) + 1
                    else:
                        prev_seq_no = float(seq_no)
                        prev_seg_len = float(seg_len)
                except IndexError:
                    pass

            for key in range(1800):
                df = df.append({
                    'run': run,
                    'second': key,
                    'retransmit': retransmits_per_sec[key],
                    'retransmit_rate': weird_division(float(retransmits_per_sec[key]), float(pkt_per_sec[key])),
                    'r_3': r_3[key],
                    'r_3_rate': weird_division(float(r_3[key]), float(pkt_per_sec[key])),
                    'r_4': r_4[key],
                    'r_4_rate': weird_division(float(r_4[key]), float(pkt_per_sec[key])),
                    'r_300': r_300[key],
                    'r_300_rate': weird_division(float(r_300[key]), float(pkt_per_sec[key])),
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s ul satcom loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_losses_def_ul_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                                multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all PCAPs to extract packet losses.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract losses")

    df_cols = [*config_cols, 'run', 'second', 'retransmit', 'retransmit_rate',
               'r_3', 'r_3_rate', 'r_4', 'r_4_rate', 'r_300', 'r_300_rate']
    if multi_process:
        df_tcp_loss_def_ul_lte = __mp_parse_slices(4, __parse_tcp_lte_losses_def_ul_from_scenario, in_dir, compression, run_type, scenarios,
                                                   df_cols, 'tcp-ul-lte', 'tcp_loss_def_ul_lte')
    else:
        df_tcp_loss_def_ul_lte = __parse_slice(__parse_tcp_lte_losses_def_ul_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                               df_cols, 'tcp-ul-lte', 'tcp_loss_def_ul_lte')

    logger.debug("Fixing tcp lte loss data types")
    df_tcp_loss_def_ul_lte = fix_dtypes(df_tcp_loss_def_ul_lte)

    logger.info("Saving tcp loss data")
    df_tcp_loss_def_ul_lte.to_pickle(
        os.path.join(out_dir, 'tcp_loss_def_ul_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_loss_def_ul_lte.csv'), 'w+') as out_file:
        df_tcp_loss_def_ul_lte.to_csv(out_file)

    return df_tcp_loss_def_ul_lte


def __parse_tcp_lte_losses_def_ul_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp lte results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s ul lte files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'retransmit',
                      'retransmit_rate', 'r_3', 'r_3_rate', 'r_4', 'r_4_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap.xz$" %
                               ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3\.pcap$" %
                              ("_pep" if pep else "",), file_name) if int(run_type) == 4 \
                else re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_client_ue3_st3\.pcap$" %
                               ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_sender_loss_ul_lte.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.src==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            compress_cmd = "xz " + file_path[:-3]
            os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_sender_loss_ul_lte.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.src==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.GSTREAMER_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            retransmits_per_sec = dict.fromkeys(range(1800), 0)
            pkt_per_sec = dict.fromkeys(range(1800), 0)
            first_row = next(results)
            initial_ts = first_row[0]
            prev_seq_no = first_row[1]
            prev_seg_len = 1
            retransmits_per_rto = {}
            tcp_rto = 0.2
            tcp_rto_300 = 0.3
            tcp_rto_3 = 1.4
            tcp_rto_4 = 3.0
            r_3 = dict.fromkeys(range(1800), 0)
            r_4 = dict.fromkeys(range(1800), 0)
            r_300 = dict.fromkeys(range(1800), 0)
            for interval in islice(results, 0, row_count, None):
                try:
                    curr_ts = interval[0]
                    seq_no = float(interval[1])
                    seg_len = interval[3]
                    duration = float(curr_ts) - float(initial_ts)
                    duration = int(duration)
                    pkt_per_sec[duration] = pkt_per_sec.get(duration, 0) + 1

                    # Check if packet is a retransmission by retransmit flag and if the
                    # sequence number repeats after a TCP RTO of 200 ms.
                    expected_seq_no = float(prev_seq_no) + float(prev_seg_len)
                    # print(seq_no, expected_seq_no, prev_seq_no, prev_seg_len)
                    # print(type(seq_no), type(expected_seq_no))
                    # print(seq_no == expected_seq_no)
                    if seq_no != expected_seq_no:
                        if seq_no not in retransmits_per_rto:
                            retransmits_per_rto[seq_no] = curr_ts
                            retransmits_per_sec[duration] = retransmits_per_sec.get(
                                duration, 0) + 1
                        else:
                            prev_ts = retransmits_per_rto.get(seq_no)
                            retransmit_interval = float(
                                curr_ts) - float(prev_ts)
                            if (retransmit_interval > tcp_rto):
                                retransmits_per_sec[duration] = retransmits_per_sec.get(
                                    duration, 0) + 1
                            if (retransmit_interval > tcp_rto_3):
                                r_3[duration] = r_3.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_4):
                                r_4[duration] = r_4.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_300):
                                r_300[duration] = r_300.get(duration, 0) + 1
                    else:
                        prev_seq_no = float(seq_no)
                        prev_seg_len = float(seg_len)
                except IndexError:
                    pass

            for key in range(1800):
                df = df.append({
                    'run': run,
                    'second': key,
                    'retransmit': retransmits_per_sec[key],
                    'retransmit_rate': weird_division(float(retransmits_per_sec[key]), float(pkt_per_sec[key])),
                    'r_3': r_3[key],
                    'r_3_rate': weird_division(float(r_3[key]), float(pkt_per_sec[key])),
                    'r_4': r_4[key],
                    'r_4_rate': weird_division(float(r_4[key]), float(pkt_per_sec[key])),
                    'r_300': r_300[key],
                    'r_300_rate': weird_division(float(r_300[key]), float(pkt_per_sec[key])),
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s ul lte loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_losses_def_dl_lte(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                                multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all PCAPs to extract packet losses.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract losses")

    df_cols = [*config_cols, 'run', 'second', 'retransmit', 'retransmit_rate',
               'r_3', 'r_3_rate', 'r_4', 'r_4_rate', 'r_300', 'r_300_rate']

    if multi_process:
        df_tcp_loss_def_dl_lte = __mp_parse_slices(4, __parse_tcp_lte_losses_def_dl_from_scenario, in_dir, compression, run_type, scenarios,
                                                   df_cols, 'tcp-dl-lte', 'tcp_loss_def_dl_lte')
    else:
        df_tcp_loss_def_dl_lte = __parse_slice(__parse_tcp_lte_losses_def_dl_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                               df_cols, 'tcp-dl-lte', 'tcp_loss_def_dl_lte')

    logger.debug("Fixing tcp lte loss data types")
    df_tcp_loss_def_dl_lte = fix_dtypes(df_tcp_loss_def_dl_lte)

    logger.info("Saving tcp loss data")
    df_tcp_loss_def_dl_lte.to_pickle(
        os.path.join(out_dir, 'tcp_loss_def_dl_lte.pkl'))
    with open(os.path.join(out_dir, 'tcp_loss_def_dl_lte.csv'), 'w+') as out_file:
        df_tcp_loss_def_dl_lte.to_csv(out_file)

    return df_tcp_loss_def_dl_lte


def __parse_tcp_lte_losses_def_dl_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp lte results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s dl lte files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'retransmit',
                      'retransmit_rate', 'r_3', 'r_3_rate', 'r_4', 'r_4_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        # temporary file name
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_sender_loss_def_dl_lte.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.dst==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.IPERF_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            if int(compression) == 1:
                compress_cmd = "xz " + file_path[:-3]
                os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_sender_loss_def_dl_lte.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.dst==" + \
                common.CLIENT_IP_LTE + " and tcp.port==" + \
                common.IPERF_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            retransmits_per_sec = dict.fromkeys(range(1800), 0)
            pkt_per_sec = dict.fromkeys(range(1800), 0)
            first_row = next(results)
            initial_ts = first_row[0]
            prev_seq_no = first_row[1]
            prev_seg_len = 1
            retransmits_per_rto = {}
            tcp_rto = 0.2
            tcp_rto_300 = 0.3
            tcp_rto_3 = 1.4
            tcp_rto_4 = 3.0
            r_3 = dict.fromkeys(range(1800), 0)
            r_4 = dict.fromkeys(range(1800), 0)
            r_300 = dict.fromkeys(range(1800), 0)
            for interval in islice(results, 0, row_count, None):
                try:
                    curr_ts = interval[0]
                    seq_no = float(interval[1])
                    seg_len = interval[3]
                    duration = float(curr_ts) - float(initial_ts)
                    duration = int(duration)
                    pkt_per_sec[duration] = pkt_per_sec.get(duration, 0) + 1

                    # Check if packet is a retransmission by retransmit flag and if the
                    # sequence number repeats after a TCP RTO of 200 ms.
                    expected_seq_no = float(prev_seq_no) + float(prev_seg_len)
                    # print(seq_no, expected_seq_no, prev_seq_no, prev_seg_len)
                    # print(type(seq_no), type(expected_seq_no))
                    # print(seq_no == expected_seq_no)
                    if seq_no != expected_seq_no:
                        if seq_no not in retransmits_per_rto:
                            retransmits_per_rto[seq_no] = curr_ts
                            retransmits_per_sec[duration] = retransmits_per_sec.get(
                                duration, 0) + 1
                        else:
                            prev_ts = retransmits_per_rto.get(seq_no)
                            retransmit_interval = float(
                                curr_ts) - float(prev_ts)
                            if (retransmit_interval > tcp_rto):
                                retransmits_per_sec[duration] = retransmits_per_sec.get(
                                    duration, 0) + 1
                            if (retransmit_interval > tcp_rto_3):
                                r_3[duration] = r_3.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_4):
                                r_4[duration] = r_4.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_300):
                                r_300[duration] = r_300.get(duration, 0) + 1
                    else:
                        prev_seq_no = float(seq_no)
                        prev_seg_len = float(seg_len)
                except IndexError:
                    pass

            for key in range(1800):
                df = df.append({
                    'run': run,
                    'second': key,
                    'retransmit': retransmits_per_sec[key],
                    'retransmit_rate': weird_division(float(retransmits_per_sec[key]), float(pkt_per_sec[key])),
                    'r_3': r_3[key],
                    'r_3_rate': weird_division(float(r_3[key]), float(pkt_per_sec[key])),
                    'r_4': r_4[key],
                    'r_4_rate': weird_division(float(r_4[key]), float(pkt_per_sec[key])),
                    'r_300': r_300[key],
                    'r_300_rate': weird_division(float(r_300[key]), float(pkt_per_sec[key])),
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s dl lte loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_losses_def_dl_sat(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_cols: List[str],
                                multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all PCAPs to extract packet losses.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing PCAPs to extract losses")

    df_cols = [*config_cols, 'run', 'second', 'retransmit', 'retransmit_rate',
               'r_3', 'r_3_rate', 'r_4', 'r_4_rate', 'r_300', 'r_300_rate']

    if multi_process:
        df_tcp_loss_def_dl_sat = __mp_parse_slices(4, __parse_tcp_sat_losses_def_dl_from_scenario, in_dir, compression, run_type, scenarios,
                                                   df_cols, 'tcp-dl-sat', 'tcp_loss_def_dl_sat')
    else:
        df_tcp_loss_def_dl_sat = __parse_slice(__parse_tcp_sat_losses_def_dl_from_scenario, in_dir, compression, run_type, [*scenarios.items()],
                                               df_cols, 'tcp-dl-sat', 'tcp_loss_def_dl_sat')

    logger.debug("Fixing tcp dl satcom loss data types")
    df_tcp_loss_def_dl_sat = fix_dtypes(df_tcp_loss_def_dl_sat)

    logger.info("Saving tcp dl loss data")
    df_tcp_loss_def_dl_sat.to_pickle(
        os.path.join(out_dir, 'tcp_loss_def_dl_sat.pkl'))
    with open(os.path.join(out_dir, 'tcp_loss_def_dl_sat.csv'), 'w+') as out_file:
        df_tcp_loss_def_dl_sat.to_csv(out_file)

    return df_tcp_loss_def_dl_sat


def __parse_tcp_sat_losses_def_dl_from_scenario(in_dir: str, compression: int, run_type: int, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp dl satcom results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s dl satcom files in %s",
                 " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'retransmit',
                      'retransmit_rate', 'r_3', 'r_3_rate', 'r_4', 'r_4_rate'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        if int(compression) == 1:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap.xz$" %
                              ("_pep" if pep else "",), file_name)
        else:
            match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_server_gw5\.pcap$" %
                              ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        if not os.path.isfile(file_path):
            return None

        parent_dir = os.path.dirname(os.path.dirname(file_path))
        processed_file_name = None

        if int(compression) == 1:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-8] + "_sender_loss_def_dl_sat.csv"

            # Decompress PCAP file using xz
            decompress_cmd = "xz -d " + file_path
            os.system(decompress_cmd)

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path[:-3] + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.dst==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.IPERF_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

            # Compress PCAP file using xz
            if int(compression) == 1:
                compress_cmd = "xz " + file_path[:-3]
                os.system(compress_cmd)
        else:
            # temporary file name
            processed_file_name = parent_dir + "/" + common.RAW_DATA_DIR + \
                "/" + file_name[:-5] + "_sender_loss_def_dl_sat.csv"

            # Analyze PCAP using tshark
            extract_pcap_cmd = "tshark -r " + file_path + \
                " -T fields -E separator=, -e frame.time_epoch -e tcp.seq -e tcp.ack -e tcp.len ip.dst==" + \
                common.CLIENT_IP_SAT + " and tcp.port==" + \
                common.IPERF_PORT + " > " + processed_file_name
            os.system(extract_pcap_cmd)

        with open(processed_file_name, 'r') as file:
            results = csv.reader(file, delimiter=',')
            if results is None:
                logger.warning("%s: '%s' has no content",
                               scenario_name, file_path)
                continue

            row_count = sum(1 for row in results)
            row_count = row_count - 1
            file.seek(0)
            results = csv.reader(file, delimiter=',')
            retransmits_per_sec = dict.fromkeys(range(1800), 0)
            pkt_per_sec = dict.fromkeys(range(1800), 0)
            first_row = next(results)
            initial_ts = first_row[0]
            prev_seq_no = first_row[1]
            prev_seg_len = 1
            retransmits_per_rto = {}
            tcp_rto = 0.2
            tcp_rto_300 = 0.3
            tcp_rto_3 = 1.4
            tcp_rto_4 = 3.0
            r_3 = dict.fromkeys(range(1800), 0)
            r_4 = dict.fromkeys(range(1800), 0)
            r_300 = dict.fromkeys(range(1800), 0)
            for interval in islice(results, 0, row_count, None):
                try:
                    curr_ts = interval[0]
                    seq_no = float(interval[1])
                    seg_len = interval[3]
                    duration = float(curr_ts) - float(initial_ts)
                    duration = int(duration)
                    pkt_per_sec[duration] = pkt_per_sec.get(duration, 0) + 1

                    # Check if packet is a retransmission by retransmit flag and if the
                    # sequence number repeats after a TCP RTO of 200 ms.
                    expected_seq_no = float(prev_seq_no) + float(prev_seg_len)
                    # print(seq_no, expected_seq_no, prev_seq_no, prev_seg_len)
                    # print(type(seq_no), type(expected_seq_no))
                    # print(seq_no == expected_seq_no)
                    if seq_no != expected_seq_no:
                        if seq_no not in retransmits_per_rto:
                            retransmits_per_rto[seq_no] = curr_ts
                            retransmits_per_sec[duration] = retransmits_per_sec.get(
                                duration, 0) + 1
                        else:
                            prev_ts = retransmits_per_rto.get(seq_no)
                            retransmit_interval = float(
                                curr_ts) - float(prev_ts)
                            if (retransmit_interval > tcp_rto):
                                retransmits_per_sec[duration] = retransmits_per_sec.get(
                                    duration, 0) + 1
                            if (retransmit_interval > tcp_rto_3):
                                r_3[duration] = r_3.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_4):
                                r_4[duration] = r_4.get(duration, 0) + 1
                            if (retransmit_interval > tcp_rto_300):
                                r_300[duration] = r_300.get(duration, 0) + 1
                    else:
                        prev_seq_no = float(seq_no)
                        prev_seg_len = float(seg_len)
                except IndexError:
                    pass

            for key in range(1800):
                df = df.append({
                    'run': run,
                    'second': key,
                    'retransmit': retransmits_per_sec[key],
                    'retransmit_rate': weird_division(float(retransmits_per_sec[key]), float(pkt_per_sec[key])),
                    'r_3': r_3[key],
                    'r_3_rate': weird_division(float(r_3[key]), float(pkt_per_sec[key])),
                    'r_4': r_4[key],
                    'r_4_rate': weird_division(float(r_4[key]), float(pkt_per_sec[key])),
                    'r_300': r_300[key],
                    'r_300_rate': weird_division(float(r_300[key]), float(pkt_per_sec[key])),
                }, ignore_index=True)

        # Remove intermediate files
        remove_cmd = "rm " + processed_file_name
        os.system(remove_cmd)

    if df.empty:
        logger.warning("%s: No tcp%s dl satcom loss data found",
                       scenario_name, " (pep)" if pep else "")

    return df


def parse_log(in_dir: str, out_dir: str, measure_type: common.MeasureType) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_runs = None
    df_stats = None
    dfs = __parse_log(in_dir, measure_type)
    if dfs is not None:
        df_runs, df_stats = dfs
    else:
        logger.warning("No logging data")

    if df_runs is None:
        df_runs = pd.DataFrame(
            columns=['name'], index=pd.TimedeltaIndex([], name='time'))
    if df_stats is None:
        df_stats = pd.DataFrame(
            columns=['cpu_load', 'ram_usage'], index=pd.TimedeltaIndex([], name='time'))

    logger.debug("Fixing log data types")
    df_runs = fix_dtypes(df_runs)
    df_stats = fix_dtypes(df_stats)

    logger.info("Saving stats data")
    df_runs.to_pickle(os.path.join(out_dir, 'runs.pkl'))
    df_stats.to_pickle(os.path.join(out_dir, 'stats.pkl'))
    with open(os.path.join(out_dir, 'runs.csv'), 'w+') as out_file:
        df_runs.to_csv(out_file)
    with open(os.path.join(out_dir, 'stats.csv'), 'w+') as out_file:
        df_stats.to_csv(out_file)

    return df_runs, df_stats


def __parse_log(in_dir: str, measure_type: common.MeasureType) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    logger.info("Parsing log file")

    path = None
    if measure_type == common.MeasureType.OPENSAND:
        path = os.path.join(in_dir, "opensand.log")
    elif measure_type == common.MeasureType.NETEM:
        path = os.path.join(in_dir, "measure.log")
    if not os.path.isfile(path):
        logger.warning("No log file found")
        return None

    runs_data = []
    stats_data = []
    start_time = None

    with open(path) as file:
        for line in file:
            if start_time is None:
                start_time = datetime.strptime(
                    ' '.join(line.split(' ', 2)[:2]), "%Y-%m-%d %H:%M:%S%z")

            match = re.match(r"^([0-9-+ :]+) \[INFO]: (.* run \d+/\d+)$", line)
            if match:
                runs_data.append({
                    'time': datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S%z") - start_time,
                    'name': match.group(2),
                })
            else:
                match = re.search(r"^([0-9-+ :]+) \[STAT]: CPU load \(1m avg\): (\d+(?:\.\d+)?), RAM usage: (\d+)MB$",
                                  line)
                if match:
                    stats_data.append({
                        'time': datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S%z") - start_time,
                        'cpu_load': match.group(2),
                        'ram_usage': match.group(3),
                    })

    runs_df = None
    if len(runs_data) > 0:
        runs_df = pd.DataFrame(runs_data)
        runs_df.set_index('time', inplace=True)

    stats_df = None
    if len(stats_data) > 0:
        stats_df = pd.DataFrame(stats_data)
        stats_df.set_index('time', inplace=True)

    return runs_df, stats_df


def __create_config_df(out_dir: str, scenarios: Dict[str, Dict]) -> pd.DataFrame:
    df_config = pd.DataFrame(data=[config for config in scenarios.values()])
    if not df_config.empty:
        df_config.set_index('name', inplace=True)
        df_config.sort_index(inplace=True)

    logger.info("Saving config data")
    df_config.to_pickle(os.path.join(out_dir, 'config.pkl'))
    with open(os.path.join(out_dir, 'config.csv'), 'w+') as out_file:
        df_config.to_csv(out_file)

    return df_config


def __parse_results_tcp_duplex_mp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                  measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:
    tasks = [
        ('tcp_gp_ul_lte', parse_tcp_goodput_ul_lte, mp.Pipe()),
        ('tcp_gp_ul_sat', parse_tcp_goodput_ul_sat, mp.Pipe()),
        ('tcp_gp_dl_lte', parse_tcp_goodput_dl_lte, mp.Pipe()),
        ('tcp_gp_dl_sat', parse_tcp_goodput_dl_sat, mp.Pipe()),
        ('tcp_rtt_ul_lte', parse_tcp_rtt_ul_lte, mp.Pipe()),
        ('tcp_rtt_ul_sat', parse_tcp_rtt_ul_sat, mp.Pipe()),
        ('tcp_rtt_dl_lte', parse_tcp_rtt_dl_lte, mp.Pipe()),
        ('tcp_rtt_dl_sat', parse_tcp_rtt_dl_sat, mp.Pipe()),
        ('tcp_owd_ul_lte', parse_tcp_owd_ul_lte, mp.Pipe()),
        ('tcp_owd_ul_sat', parse_tcp_owd_ul_sat, mp.Pipe()),
        ('tcp_owd_dl_lte', parse_tcp_owd_dl_lte, mp.Pipe()),
        ('tcp_owd_dl_sat', parse_tcp_owd_dl_sat, mp.Pipe()),
        ('tcp_loss_ul_lte', parse_tcp_losses_ul_lte, mp.Pipe()),
        ('tcp_loss_ul_sat', parse_tcp_losses_ul_sat, mp.Pipe()),
        ('tcp_loss_dl_lte', parse_tcp_losses_dl_lte, mp.Pipe()),
        ('tcp_loss_dl_sat', parse_tcp_losses_dl_sat, mp.Pipe()),
        ('tcp_loss_def_ul_lte', parse_tcp_losses_def_ul_lte, mp.Pipe()),
        ('tcp_loss_def_ul_sat', parse_tcp_losses_def_ul_sat, mp.Pipe()),
        ('tcp_loss_def_dl_lte', parse_tcp_losses_def_dl_lte, mp.Pipe()),
        ('tcp_loss_def_dl_sat', parse_tcp_losses_def_dl_sat, mp.Pipe()),
    ]

    processes = [
        mp.Process(target=__mp_function_wrapper, name=name,
                   args=(func, client_con, in_dir, out_dir, scenarios, config_columns, True))
        for name, func, (_, client_con) in tasks
    ]

    logger.info("Starting parsing processes")

    for p in processes:
        p.start()

    # work on main thread
    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    # collect data
    parsed_results = {
        name: parent_con.recv()
        for name, _, (parent_con, _) in tasks
    }

    # wait for child processes to finish
    for p in processes:
        logger.debug("Waiting for process %s", p.name)
        p.join()
    logger.info("Parsing processes done")

    parsed_results['ping_raw'], parsed_results['ping_summary'] = parsed_results['ping']
    del parsed_results['ping']

    parsed_results['config'] = df_config
    parsed_results['stats'] = df_stats
    parsed_results['runs'] = df_runs

    return parsed_results


def __parse_results_tcp_duplex_sp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                  measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:

    df_tcp_gp_ul_lte = parse_tcp_goodput_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_ul_sat = parse_tcp_goodput_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_dl_lte = parse_tcp_goodput_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_dl_sat = parse_tcp_goodput_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_rtt_ul_lte = parse_tcp_rtt_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_ul_sat = parse_tcp_rtt_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_dl_lte = parse_tcp_rtt_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_dl_sat = parse_tcp_rtt_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_owd_dl_lte = parse_tcp_owd_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_dl_sat = parse_tcp_owd_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_ul_lte = parse_tcp_owd_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_ul_sat = parse_tcp_owd_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_loss_ul_lte = parse_tcp_losses_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_ul_sat = parse_tcp_losses_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_dl_lte = parse_tcp_losses_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_dl_sat = parse_tcp_losses_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_loss_def_ul_lte = parse_tcp_losses_def_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_def_ul_sat = parse_tcp_losses_def_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_def_dl_lte = parse_tcp_losses_def_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_def_dl_sat = parse_tcp_losses_def_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    return {
        'config': df_config,
        'tcp_gp_ul_lte': df_tcp_gp_ul_lte,
        'tcp_gp_ul_sat': df_tcp_gp_ul_sat,
        'tcp_gp_dl_lte': df_tcp_gp_dl_lte,
        'tcp_gp_dl_sat': df_tcp_gp_dl_sat,
        'tcp_rtt_ul_lte': df_tcp_rtt_ul_lte,
        'tcp_rtt_ul_sat': df_tcp_rtt_ul_sat,
        'tcp_rtt_dl_lte': df_tcp_rtt_dl_lte,
        'tcp_rtt_dl_sat': df_tcp_rtt_dl_sat,
        'tcp_owd_dl_lte': df_tcp_owd_dl_lte,
        'tcp_owd_dl_sat': df_tcp_owd_dl_sat,
        'tcp_owd_ul_lte': df_tcp_owd_ul_lte,
        'tcp_owd_ul_sat': df_tcp_owd_ul_sat,
        'tcp_loss_ul_lte': df_tcp_loss_ul_lte,
        'tcp_loss_ul_sat': df_tcp_loss_ul_sat,
        'tcp_loss_dl_lte': df_tcp_loss_dl_lte,
        'tcp_loss_dl_sat': df_tcp_loss_dl_sat,
        'tcp_loss_def_ul_lte': df_tcp_loss_def_ul_lte,
        'tcp_loss_def_ul_sat': df_tcp_loss_def_ul_sat,
        'tcp_loss_def_dl_lte': df_tcp_loss_def_dl_lte,
        'tcp_loss_def_dl_sat': df_tcp_loss_def_dl_sat,
        'stats': df_stats,
        'runs': df_runs,
    }


def __parse_results_mptcp_duplex_mp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                    measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:
    tasks = [
        ('mptcp_gp_dl', parse_mptcp_goodput_dl, mp.Pipe()),
        ('mptcp_gp_ul', parse_mptcp_goodput_ul, mp.Pipe()),
        ('mptcp_rtt_dl', parse_mptcp_rtt_dl, mp.Pipe()),
        ('mptcp_rtt_ul', parse_mptcp_rtt_ul, mp.Pipe()),
        ('mptcp_owd_dl', parse_mptcp_owd_dl, mp.Pipe()),
        ('mptcp_owd_ul', parse_mptcp_owd_ul, mp.Pipe()),
        ('mptcp_loss_dl', parse_mptcp_loss_dl, mp.Pipe()),
        ('mptcp_loss_ul', parse_mptcp_loss_ul, mp.Pipe()),
        ('mptcp_path_util_dl', parse_mptcp_path_util_dl, mp.Pipe()),
        ('mptcp_path_util_ul', parse_mptcp_path_util_ul, mp.Pipe()),
        ('mptcp_ofo_queue_dl', parse_mptcp_ofo_queue_dl, mp.Pipe()),
        ('mptcp_ofo_queue_ul', parse_mptcp_ofo_queue_ul, mp.Pipe()),
    ]

    processes = [
        mp.Process(target=__mp_function_wrapper, name=name,
                   args=(func, client_con, in_dir, out_dir, scenarios, config_columns, True))
        for name, func, (_, client_con) in tasks
    ]

    logger.info("Starting parsing processes")

    for p in processes:
        p.start()

    # work on main thread
    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    # collect data
    parsed_results = {
        name: parent_con.recv()
        for name, _, (parent_con, _) in tasks
    }

    # wait for child processes to finish
    for p in processes:
        logger.debug("Waiting for process %s", p.name)
        p.join()
    logger.info("Parsing processes done")

    parsed_results['ping_raw'], parsed_results['ping_summary'] = parsed_results['ping']
    del parsed_results['ping']

    parsed_results['config'] = df_config
    parsed_results['stats'] = df_stats
    parsed_results['runs'] = df_runs

    return parsed_results


def __parse_results_mptcp_duplex_sp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                    measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:

    # df_mptcp_gp_dl = parse_mptcp_goodput_dl(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)
    # df_mptcp_gp_ul = parse_mptcp_goodput_ul(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)

    # df_mptcp_rtt_dl = parse_mptcp_rtt_dl(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)
    # df_mptcp_rtt_ul = parse_mptcp_rtt_ul(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)

    # df_mptcp_owd_dl = parse_mptcp_owd_dl(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)
    # df_mptcp_owd_ul = parse_mptcp_owd_ul(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)

    # df_mptcp_loss_dl = parse_mptcp_loss_dl(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)
    # df_mptcp_loss_ul = parse_mptcp_loss_ul(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)

    # df_mptcp_path_util_dl = parse_mptcp_path_util_dl(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)
    # df_mptcp_path_util_ul = parse_mptcp_path_util_ul(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)

    # df_mptcp_ofo_queue_ul = parse_mptcp_ofo_queue_ul(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)
    # df_mptcp_ofo_queue_dl = parse_mptcp_ofo_queue_dl(
    #     in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    return {
        'config': df_config,
        # 'mptcp_gp_dl': df_mptcp_gp_dl,
        # 'mptcp_gp_ul': df_mptcp_gp_ul,
        # 'mptcp_rtt_dl': df_mptcp_rtt_dl,
        # 'mptcp_rtt_ul': df_mptcp_rtt_ul,
        # 'mptcp_owd_dl': df_mptcp_owd_dl,
        # 'mptcp_owd_ul': df_mptcp_owd_ul,
        # 'mptcp_loss_dl': df_mptcp_loss_dl,
        # 'mptcp_loss_ul': df_mptcp_loss_ul,
        # 'mptcp_path_util_dl': df_mptcp_path_util_dl,
        # 'mptcp_path_util_ul': df_mptcp_path_util_ul,
        # 'mptcp_ofo_queue_ul': df_mptcp_ofo_queue_ul,
        # 'mptcp_ofo_queue_dl': df_mptcp_ofo_queue_dl,
        'stats': df_stats,
        'runs': df_runs,
    }


def __parse_results_mptcp_tcp_duplex_mp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                        measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:
    tasks = [
        ('mptcp_gp_dl', parse_mptcp_goodput_dl, mp.Pipe()),
        ('mptcp_gp_ul', parse_mptcp_goodput_ul, mp.Pipe()),
        ('mptcp_rtt_dl', parse_mptcp_rtt_dl, mp.Pipe()),
        ('mptcp_rtt_ul', parse_mptcp_rtt_ul, mp.Pipe()),
        ('mptcp_owd_dl', parse_mptcp_owd_dl, mp.Pipe()),
        ('mptcp_owd_ul', parse_mptcp_owd_ul, mp.Pipe()),
        ('mptcp_loss_dl', parse_mptcp_loss_dl, mp.Pipe()),
        ('mptcp_loss_ul', parse_mptcp_loss_ul, mp.Pipe()),
        ('mptcp_path_util_dl', parse_mptcp_path_util_dl, mp.Pipe()),
        ('mptcp_path_util_ul', parse_mptcp_path_util_ul, mp.Pipe()),
        ('mptcp_ofo_queue_dl', parse_mptcp_ofo_queue_dl, mp.Pipe()),
        ('mptcp_ofo_queue_ul', parse_mptcp_ofo_queue_ul, mp.Pipe()),
        ('tcp_gp_ul_lte', parse_tcp_goodput_ul_lte, mp.Pipe()),
        ('tcp_gp_ul_sat', parse_tcp_goodput_ul_sat, mp.Pipe()),
        ('tcp_gp_dl_lte', parse_tcp_goodput_dl_lte, mp.Pipe()),
        ('tcp_gp_dl_sat', parse_tcp_goodput_dl_sat, mp.Pipe()),
        ('tcp_rtt_ul_lte', parse_tcp_rtt_ul_lte, mp.Pipe()),
        ('tcp_rtt_ul_sat', parse_tcp_rtt_ul_sat, mp.Pipe()),
        ('tcp_rtt_dl_lte', parse_tcp_rtt_dl_lte, mp.Pipe()),
        ('tcp_rtt_dl_sat', parse_tcp_rtt_dl_sat, mp.Pipe()),
        ('tcp_owd_ul_lte', parse_tcp_owd_ul_lte, mp.Pipe()),
        ('tcp_owd_ul_sat', parse_tcp_owd_ul_sat, mp.Pipe()),
        ('tcp_owd_dl_lte', parse_tcp_owd_dl_lte, mp.Pipe()),
        ('tcp_owd_dl_sat', parse_tcp_owd_dl_sat, mp.Pipe()),
        ('tcp_loss_ul_lte', parse_tcp_losses_ul_lte, mp.Pipe()),
        ('tcp_loss_ul_sat', parse_tcp_losses_ul_sat, mp.Pipe()),
        ('tcp_loss_dl_lte', parse_tcp_losses_dl_lte, mp.Pipe()),
        ('tcp_loss_dl_sat', parse_tcp_losses_dl_sat, mp.Pipe()),
        ('tcp_loss_def_ul_lte', parse_tcp_losses_def_ul_lte, mp.Pipe()),
        ('tcp_loss_def_ul_sat', parse_tcp_losses_def_ul_sat, mp.Pipe()),
        ('tcp_loss_def_dl_lte', parse_tcp_losses_def_dl_lte, mp.Pipe()),
        ('tcp_loss_def_dl_sat', parse_tcp_losses_def_dl_sat, mp.Pipe()),
    ]

    processes = [
        mp.Process(target=__mp_function_wrapper, name=name,
                   args=(func, client_con, in_dir, out_dir, scenarios, config_columns, True))
        for name, func, (_, client_con) in tasks
    ]

    logger.info("Starting parsing processes")

    for p in processes:
        p.start()

    # work on main thread
    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    # collect data
    parsed_results = {
        name: parent_con.recv()
        for name, _, (parent_con, _) in tasks
    }

    # wait for child processes to finish
    for p in processes:
        logger.debug("Waiting for process %s", p.name)
        p.join()
    logger.info("Parsing processes done")

    parsed_results['ping_raw'], parsed_results['ping_summary'] = parsed_results['ping']
    del parsed_results['ping']

    parsed_results['config'] = df_config
    parsed_results['stats'] = df_stats
    parsed_results['runs'] = df_runs

    return parsed_results


def __parse_results_mptcp_tcp_duplex_sp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                        measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:

    df_mptcp_gp_dl = parse_mptcp_goodput_dl(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_mptcp_gp_ul = parse_mptcp_goodput_ul(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_ul_lte = parse_tcp_goodput_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_ul_sat = parse_tcp_goodput_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_dl_lte = parse_tcp_goodput_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_dl_sat = parse_tcp_goodput_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_mptcp_rtt_dl = parse_mptcp_rtt_dl(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_mptcp_rtt_ul = parse_mptcp_rtt_ul(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_ul_lte = parse_tcp_rtt_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_ul_sat = parse_tcp_rtt_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_dl_lte = parse_tcp_rtt_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_dl_sat = parse_tcp_rtt_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_mptcp_owd_dl = parse_mptcp_owd_dl(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_mptcp_owd_ul = parse_mptcp_owd_ul(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_dl_lte = parse_tcp_owd_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_dl_sat = parse_tcp_owd_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_ul_lte = parse_tcp_owd_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_ul_sat = parse_tcp_owd_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_mptcp_loss_dl = parse_mptcp_loss_dl(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_mptcp_loss_ul = parse_mptcp_loss_ul(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_ul_lte = parse_tcp_losses_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_ul_sat = parse_tcp_losses_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_dl_lte = parse_tcp_losses_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_dl_sat = parse_tcp_losses_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_loss_def_ul_lte = parse_tcp_losses_def_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_def_ul_sat = parse_tcp_losses_def_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_def_dl_lte = parse_tcp_losses_def_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_def_dl_sat = parse_tcp_losses_def_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_mptcp_path_util_dl = parse_mptcp_path_util_dl(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_mptcp_path_util_ul = parse_mptcp_path_util_ul(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_mptcp_ofo_queue_ul = parse_mptcp_ofo_queue_ul(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_mptcp_ofo_queue_dl = parse_mptcp_ofo_queue_dl(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    return {
        'config': df_config,
        'mptcp_gp_dl': df_mptcp_gp_dl,
        'mptcp_gp_ul': df_mptcp_gp_ul,
        'tcp_gp_ul_lte': df_tcp_gp_ul_lte,
        'tcp_gp_ul_sat': df_tcp_gp_ul_sat,
        'tcp_gp_dl_lte': df_tcp_gp_dl_lte,
        'tcp_gp_dl_sat': df_tcp_gp_dl_sat,
        'mptcp_rtt_dl': df_mptcp_rtt_dl,
        'mptcp_rtt_ul': df_mptcp_rtt_ul,
        'mptcp_owd_dl': df_mptcp_owd_dl,
        'mptcp_owd_ul': df_mptcp_owd_ul,
        'tcp_owd_dl_lte': df_tcp_owd_dl_lte,
        'tcp_owd_dl_sat': df_tcp_owd_dl_sat,
        'tcp_owd_ul_lte': df_tcp_owd_ul_lte,
        'tcp_owd_ul_sat': df_tcp_owd_ul_sat,
        'tcp_rtt_ul_lte': df_tcp_rtt_ul_lte,
        'tcp_rtt_ul_sat': df_tcp_rtt_ul_sat,
        'tcp_rtt_dl_lte': df_tcp_rtt_dl_lte,
        'tcp_rtt_dl_sat': df_tcp_rtt_dl_sat,
        'mptcp_loss_dl': df_mptcp_loss_dl,
        'mptcp_loss_ul': df_mptcp_loss_ul,
        'tcp_loss_ul_lte': df_tcp_loss_ul_lte,
        'tcp_loss_ul_sat': df_tcp_loss_ul_sat,
        'tcp_loss_dl_lte': df_tcp_loss_dl_lte,
        'tcp_loss_dl_sat': df_tcp_loss_dl_sat,
        'tcp_loss_def_ul_lte': df_tcp_loss_def_ul_lte,
        'tcp_loss_def_ul_sat': df_tcp_loss_def_ul_sat,
        'tcp_loss_def_dl_lte': df_tcp_loss_def_dl_lte,
        'tcp_loss_def_dl_sat': df_tcp_loss_def_dl_sat,
        'mptcp_path_util_dl': df_mptcp_path_util_dl,
        'mptcp_path_util_ul': df_mptcp_path_util_ul,
        'mptcp_ofo_queue_ul': df_mptcp_ofo_queue_ul,
        'mptcp_ofo_queue_dl': df_mptcp_ofo_queue_dl,
        'stats': df_stats,
        'runs': df_runs,
    }


def __parse_results_sptcp_duplex_lte_mp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                        measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:
    tasks = [
        ('tcp_gp_ul_lte', parse_tcp_goodput_ul_lte, mp.Pipe()),
        ('tcp_gp_dl_lte', parse_tcp_goodput_dl_lte, mp.Pipe()),
        ('tcp_rtt_ul_lte', parse_tcp_rtt_ul_lte, mp.Pipe()),
        ('tcp_rtt_dl_lte', parse_tcp_rtt_dl_lte, mp.Pipe()),
        ('tcp_owd_ul_lte', parse_tcp_owd_ul_lte, mp.Pipe()),
        ('tcp_owd_dl_lte', parse_tcp_owd_dl_lte, mp.Pipe()),
        ('tcp_loss_ul_lte', parse_tcp_losses_ul_lte, mp.Pipe()),
        ('tcp_loss_dl_lte', parse_tcp_losses_dl_lte, mp.Pipe()),
        ('tcp_loss_def_ul_lte', parse_tcp_losses_def_ul_lte, mp.Pipe()),
        ('tcp_loss_def_dl_lte', parse_tcp_losses_def_dl_lte, mp.Pipe()),
    ]

    processes = [
        mp.Process(target=__mp_function_wrapper, name=name,
                   args=(func, client_con, in_dir, out_dir, scenarios, config_columns, True))
        for name, func, (_, client_con) in tasks
    ]

    logger.info("Starting parsing processes")

    for p in processes:
        p.start()

    # work on main thread
    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    # collect data
    parsed_results = {
        name: parent_con.recv()
        for name, _, (parent_con, _) in tasks
    }

    # wait for child processes to finish
    for p in processes:
        logger.debug("Waiting for process %s", p.name)
        p.join()
    logger.info("Parsing processes done")

    parsed_results['ping_raw'], parsed_results['ping_summary'] = parsed_results['ping']
    del parsed_results['ping']

    parsed_results['config'] = df_config
    parsed_results['stats'] = df_stats
    parsed_results['runs'] = df_runs

    return parsed_results


def __parse_results_sptcp_duplex_lte_sp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                        measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:

    df_tcp_gp_ul_lte = parse_tcp_goodput_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_dl_lte = parse_tcp_goodput_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_rtt_ul_lte = parse_tcp_rtt_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_dl_lte = parse_tcp_rtt_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_owd_dl_lte = parse_tcp_owd_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_ul_lte = parse_tcp_owd_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_loss_ul_lte = parse_tcp_losses_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_dl_lte = parse_tcp_losses_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_loss_def_ul_lte = parse_tcp_losses_def_ul_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_def_dl_lte = parse_tcp_losses_def_dl_lte(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    return {
        'config': df_config,
        'tcp_gp_ul_lte': df_tcp_gp_ul_lte,
        'tcp_gp_dl_lte': df_tcp_gp_dl_lte,
        'tcp_rtt_ul_lte': df_tcp_rtt_ul_lte,
        'tcp_rtt_dl_lte': df_tcp_rtt_dl_lte,
        'tcp_owd_dl_lte': df_tcp_owd_dl_lte,
        'tcp_owd_ul_lte': df_tcp_owd_ul_lte,
        'tcp_loss_ul_lte': df_tcp_loss_ul_lte,
        'tcp_loss_dl_lte': df_tcp_loss_dl_lte,
        'tcp_loss_def_ul_lte': df_tcp_loss_def_ul_lte,
        'tcp_loss_def_dl_lte': df_tcp_loss_def_dl_lte,
        'stats': df_stats,
        'runs': df_runs,
    }


def __parse_results_sptcp_duplex_sat_mp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                        measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:
    tasks = [
        ('tcp_gp_ul_sat', parse_tcp_goodput_ul_sat, mp.Pipe()),
        ('tcp_gp_dl_sat', parse_tcp_goodput_dl_sat, mp.Pipe()),
        ('tcp_rtt_ul_sat', parse_tcp_rtt_ul_sat, mp.Pipe()),
        ('tcp_rtt_dl_sat', parse_tcp_rtt_dl_sat, mp.Pipe()),
        ('tcp_owd_ul_sat', parse_tcp_owd_ul_sat, mp.Pipe()),
        ('tcp_owd_dl_sat', parse_tcp_owd_dl_sat, mp.Pipe()),
        ('tcp_loss_ul_sat', parse_tcp_losses_ul_sat, mp.Pipe()),
        ('tcp_loss_dl_sat', parse_tcp_losses_dl_sat, mp.Pipe()),
        ('tcp_loss_def_ul_sat', parse_tcp_losses_def_ul_sat, mp.Pipe()),
        ('tcp_loss_def_dl_sat', parse_tcp_losses_def_dl_sat, mp.Pipe()),
    ]

    processes = [
        mp.Process(target=__mp_function_wrapper, name=name,
                   args=(func, client_con, in_dir, out_dir, scenarios, config_columns, True))
        for name, func, (_, client_con) in tasks
    ]

    logger.info("Starting parsing processes")

    for p in processes:
        p.start()

    # work on main thread
    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    # collect data
    parsed_results = {
        name: parent_con.recv()
        for name, _, (parent_con, _) in tasks
    }

    # wait for child processes to finish
    for p in processes:
        logger.debug("Waiting for process %s", p.name)
        p.join()
    logger.info("Parsing processes done")

    parsed_results['ping_raw'], parsed_results['ping_summary'] = parsed_results['ping']
    del parsed_results['ping']

    parsed_results['config'] = df_config
    parsed_results['stats'] = df_stats
    parsed_results['runs'] = df_runs

    return parsed_results


def __parse_results_sptcp_duplex_sat_sp(in_dir: str, out_dir: str, compression: int, run_type: int, scenarios: Dict[str, Dict], config_columns: List[str],
                                        measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:

    df_tcp_gp_ul_sat = parse_tcp_goodput_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_gp_dl_sat = parse_tcp_goodput_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_rtt_ul_sat = parse_tcp_rtt_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_rtt_dl_sat = parse_tcp_rtt_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_owd_dl_sat = parse_tcp_owd_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_owd_ul_sat = parse_tcp_owd_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_loss_ul_sat = parse_tcp_losses_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_dl_sat = parse_tcp_losses_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_tcp_loss_def_ul_sat = parse_tcp_losses_def_ul_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)
    df_tcp_loss_def_dl_sat = parse_tcp_losses_def_dl_sat(
        in_dir, out_dir, compression, run_type, scenarios, config_columns)

    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    return {
        'config': df_config,
        'tcp_gp_ul_sat': df_tcp_gp_ul_sat,
        'tcp_gp_dl_sat': df_tcp_gp_dl_sat,
        'tcp_rtt_ul_sat': df_tcp_rtt_ul_sat,
        'tcp_rtt_dl_sat': df_tcp_rtt_dl_sat,
        'tcp_owd_dl_sat': df_tcp_owd_dl_sat,
        'tcp_owd_ul_sat': df_tcp_owd_ul_sat,
        'tcp_loss_ul_sat': df_tcp_loss_ul_sat,
        'tcp_loss_dl_sat': df_tcp_loss_dl_sat,
        'tcp_loss_def_ul_sat': df_tcp_loss_def_ul_sat,
        'tcp_loss_def_dl_sat': df_tcp_loss_def_dl_sat,
        'stats': df_stats,
        'runs': df_runs,
    }


def decompress_logs(in_dir: str, scenario_name: List[Tuple[str, Dict]], pep: bool = False):
    for folder, config in scenario_name:
        for pep in (False, True):
            for file_name in os.listdir(os.path.join(in_dir, folder)):
                path = os.path.join(in_dir, folder)
                file_path = os.path.join(in_dir, folder, file_name)
                if not os.path.isfile(file_path):
                    continue
                match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_" %
                                  ("_pep" if pep else "",), file_name)
                if not match:
                    continue

                logger.debug("%s: Decompressing '%s'", folder, file_name)

                # Decompress PCAP file using xz
                decompress_cmd = "xz -T0 -d " + file_path
                os.system(decompress_cmd)


def compress_logs(in_dir: str, scenario_name: List[Tuple[str, Dict]], pep: bool = False):
    for folder, config in scenario_name:
        for pep in (False, True):
            for file_name in os.listdir(os.path.join(in_dir, folder)):
                path = os.path.join(in_dir, folder)
                file_path = os.path.join(in_dir, folder, file_name)
                if not os.path.isfile(file_path):
                    continue
                match = re.search(r"^tcp_iperf_duplex%s_(\d+)_dump_" %
                                  ("_pep" if pep else "",), file_name)
                if not match:
                    continue

                logger.debug("%s: Compressing '%s'", folder, file_name)

                # Compress PCAP file using xz
                compress_cmd = "xz -T0 " + file_path
                os.system(compress_cmd)


def parse_results(in_dir: str, out_dir: str, multi_process: bool = False, run_type: int = 2, compression: int = 0,
                  ) -> Tuple[common.MeasureType, Dict, Dict[str, pd.DataFrame]]:
    logger.info("Parsing measurement results in '%s'", in_dir)

    # read scenarios
    logger.info("Reading scenarios")
    scenarios = {}
    for folder_name in list_result_folders(in_dir):
        logger.debug("Parsing config in %s", folder_name)
        scenarios[folder_name] = __read_config_from_scenario(
            in_dir, folder_name)

    if len(scenarios) == 0:
        print("Failed to parse results, no scenarios found")
        sys.exit(4)
    logger.info("Found %d scenarios to parse", len(scenarios))

    # create output folder
    raw_out_dir = os.path.join(out_dir, common.RAW_DATA_DIR)
    if not os.path.exists(raw_out_dir):
        os.mkdir(raw_out_dir)

    measure_type = detect_measure_type(in_dir, out_dir)
    auto_detect = parse_auto_detect(in_dir, out_dir)

    # decompress logs
    if int(compression) == 0:
        decompress_logs(in_dir, [*scenarios.items()])

    # prepare columns
    config_columns = ['protocol', 'pep', 'sat', 'prime']
    if measure_type == common.MeasureType.NETEM:
        config_columns.extend(['rate', 'loss', 'queue'])
    elif measure_type == common.MeasureType.OPENSAND:
        config_columns.extend(
            ['attenuation', 'ccs', 'tbs', 'qbs', 'ubs', 'mp_sched', 'mp_cc'])

    if int(run_type) == 0:
        parse_func = __parse_results_tcp_duplex_mp if multi_process else __parse_results_tcp_duplex_sp
    elif int(run_type) == 1:
        parse_func = __parse_results_mptcp_duplex_mp if multi_process else __parse_results_mptcp_duplex_sp
    elif int(run_type) == 2:
        parse_func = __parse_results_mptcp_tcp_duplex_mp if multi_process else __parse_results_mptcp_tcp_duplex_sp
    elif int(run_type) == 3:
        parse_func = __parse_results_sptcp_duplex_sat_mp if multi_process else __parse_results_sptcp_duplex_sat_sp
    elif int(run_type) == 4:
        parse_func = __parse_results_sptcp_duplex_lte_mp if multi_process else __parse_results_sptcp_duplex_lte_sp

    parsed_results = parse_func(
        in_dir, raw_out_dir, compression, run_type, scenarios, config_columns, measure_type)

    # compress logs
    if int(compression) == 0:
        compress_logs(in_dir, [*scenarios.items()])

    return measure_type, auto_detect, parsed_results


def load_parsed_results(in_dir: str) -> Tuple[common.MeasureType, Dict, Dict[str, pd.DataFrame]]:
    logger.info("Loading parsed results from %s", in_dir)

    measure_type = None
    auto_detect = {}
    parsed_results = {}

    # read measure_type and auto_detect
    for file_name in os.listdir(in_dir):
        file_path = os.path.join(in_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if file_name == common.TYPE_FILE:
            logger.debug("Reading measure type")
            with open(file_path, 'r') as file:
                measure_type_str = file.readline(64)
                measure_type = common.MeasureType.from_name(measure_type_str)
                logger.debug("Read measure type str '%s' resulting in %s",
                             measure_type_str, str(measure_type))
            continue

        if file_name == common.AUTO_DETECT_FILE:
            logger.debug("Reading auto detect file")
            with open(file_path, 'r') as file:
                auto_detect = {key: value for key, value in [
                    line.split('=', 1) for line in file.readlines()]}

    # read data frames
    raw_dir = os.path.join(in_dir, common.RAW_DATA_DIR)
    for file_name in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        match = re.match(r"^(.*)\.pkl$", file_name)
        if not match:
            continue

        logger.debug("Loading %s", file_name)
        parsed_results[match.group(1)] = pd.read_pickle(file_path)

    return measure_type, auto_detect, parsed_results
