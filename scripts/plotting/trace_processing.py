import argparse
import pandas


def parse_csv(file_name, counter_num, index, output):
    df = pandas.read_csv(file_name)
    df.drop(df.columns[2:], axis=1, inplace=True)
    df.drop(df.tail(1).index, inplace=True)
    df = df.iloc[index-1::counter_num, :]
    df.to_csv(output, sep=',', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default="",
                        help='csv to read from')
    parser.add_argument('--num-counters', type=int, default='',
                        help='number of perf counters collected')
    parser.add_argument('--counter-index', type=int, default='',
                        help='Counter you wish to extract')
    parser.add_argument('--output-file', type=str, default='',
                        help='Name of output file')

    args = parser.parse_args()
    input_file = args.input_file
    num_counters = args.num_counters
    counter_index = args.counter_index
    output_file = args.output_file

    # Remove empty lines at beginning
    with open(input_file, 'r+') as fp:
        lines = fp.readlines()
        fp.seek(0)
        fp.writelines('time, value,,,,,,\n')
        fp.writelines(lines[2:])

    parse_csv(input_file, num_counters, counter_index, output_file)
