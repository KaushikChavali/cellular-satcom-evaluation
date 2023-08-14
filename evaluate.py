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

import getopt
import multiprocessing as mp
import os
import sys

import analyze
import parse
from common import Mode, logger, setup_logger


def usage(name):
    print(
        "Usage: %s -i <input> -o <output>\n"
        "\n"
        "-a, --analyze       Analyze previously parsed results\n"
        "-d, --auto-detect   Try to automatically configure analysis from input\n"
        "-h, --help          Print this help message\n"
        "-i, --input=<dir>   Input directory to read the measurement results from\n"
        "-m, --multi-process Use multiple processes while parsing and analyzing results\n"
        "-o, --output=<dir>  Output directory to put the parsed results and graphs to\n"
        "-p, --parse         Parse only and skip analysis\n"
        "-r, --run-type      0: TCP Duplex\n\
                    1: MPTCP Duplex\n\
                    2: MPTCP+TCP Duplex (default)\n\
                    3: SPTCP Duplex over SAT\n\
                    4: SPTCP Duplex over LTE\n"
        "-c, --compression   0: Decompress log files before processing [faster] (requires additional storage)\n\
                    1: Decompress log files on-the-fly (requires additional computation)\n"
        "" % name
    )


def parse_args(name, argv):
    in_dir = None
    out_dir = None
    auto_detect = False
    multi_process = False
    mode = Mode.ALL
    run_type = 2
    compression = 0

    try:
        opts, args = getopt.getopt(argv, "ac:dhi:mo:pr:", ["analyze", "auto-detect", "help", "input=", "multi-process",
                                                       "output=", "parse", "run-type=", "compression="])
    except getopt.GetoptError:
        print("parse.py -i <input_dir> -o <output_dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-a", "analyze"):
            mode = Mode.ANALYZE
        elif opt in ("-d", "--auto-detect"):
            auto_detect = True
        elif opt in ("-h", "--help"):
            usage(name)
            sys.exit(0)
        elif opt in ("-i", "--input"):
            in_dir = arg
        elif opt in ("-m", "--multi-process"):
            multi_process = True
        elif opt in ("-o", "--output"):
            out_dir = arg
        elif opt in ("-p", "parse"):
            mode = Mode.PARSE
        elif opt in ("-r", "--run-type"):
            run_type = arg
        elif opt in ("-c", "--compression"):
            compression = arg

    if in_dir is None:
        print("No input directory specified")
        print("%s -h for help", name)
        sys.exit(1)
    if out_dir is None:
        if mode == Mode.ANALYZE:
            out_dir = in_dir
        else:
            print("No output directory specified")
            print("%s -h for help" % name)
            sys.exit(1)

    return mode, in_dir, out_dir, auto_detect, multi_process, run_type, compression


def main(name, argv):
    setup_logger()
    mp.current_process().name = "main"

    mode, in_dir, out_dir, do_auto_detect, multi_process, run_type, compression = parse_args(name, argv)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(out_dir):
        logger.error("Output directory is not a directory!")
        sys.exit(1)

    measure_type = None
    auto_detect = None
    parsed_results = None

    if mode.do_parse():
        logger.info("Starting parsing")
        measure_type, auto_detect, parsed_results = parse.parse_results(in_dir, out_dir, multi_process=multi_process, run_type=run_type, compression=compression)
        logger.info("Parsing done")

    if mode.do_analyze():
        if parsed_results is None:
            measure_type, auto_detect, parsed_results = parse.load_parsed_results(in_dir)

        if do_auto_detect:
            if 'MEASURE_TIME' in auto_detect:
                analyze.GRAPH_PLOT_SECONDS = float(auto_detect['MEASURE_TIME'])
                logger.debug("Detected GRAPH_PLOT_SECONDS as %f", analyze.GRAPH_PLOT_SECONDS)
            if 'REPORT_INTERVAL' in auto_detect:
                analyze.GRAPH_X_BUCKET = float(auto_detect['REPORT_INTERVAL'])
                logger.debug("Detected GRAPH_X_BUCKET as %f", analyze.GRAPH_X_BUCKET)

        logger.info("Starting analysis")
        analyze.analyze_all(parsed_results, measure_type, out_dir, multi_process=multi_process)
        logger.info("Analysis done")


if __name__ == '__main__':
    main(sys.argv[0], sys.argv[1:])
