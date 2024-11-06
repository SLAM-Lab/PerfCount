import argparse
import pandas

def parse_csv(input_list, index, output):
    # Using readlines()
    input_file= open(input_list, 'r')
    csv_files = input_file.readlines()

    merged_df = pandas.read_csv('blank.csv')
    initial_read = 0

    for csv in csv_files:
        csv = csv.strip()
        # read the csv file
        df1 = pandas.read_csv(csv)
        df1['time'] = df1['time'].astype(float).round(2) #.round({'time':2})
        # Remove unused columns
        df1.drop(df1.columns[4:], axis=1, inplace=True)

        # Remove unusued intermediate column
        df1.drop(df1.columns[2], axis=1, inplace=True)

        # Get same counter
        df3 = df1.iloc[index-0::2, :]
        counter_name = df3.iloc[index-0][2]
        df3 = df3.rename(columns={'value': counter_name})
        df3.drop(df3.columns[2], axis=1, inplace=True)
        df3 = df3.round({'time':2})

        df4 = df1.iloc[index-1::2, :]
        counter_name = df4.iloc[index-0][2]
        df4 = df4.rename(columns={'value': counter_name})
        df4.drop(df4.columns[2], axis=1, inplace=True)
        df4 = df4.round({'time':2})

        df7 = pandas.merge(df3, df4, on = 'time', how='inner')
        df7 = df7.round({'time':2})

        if initial_read == 0:
            merged_df = pandas.merge(merged_df, df7, on = 'time', how = 'inner')
            initial_read = 1
        else:
            df9 = df9.drop(columns=['instructions'])
            merged_df = pandas.merge(merged_df, df7, on = 'time', how = 'inner')


        merged_df.to_csv(output, sep=',', index=False)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-list', type=str, default="",
                        help='csv to read from')
    parser.add_argument('--num-counters', type=int, default='',
                        help='number of perf counters collected')
    parser.add_argument('--output', type=str, default='',
                        help='Name of output file')

    args = parser.parse_args()
    input_list = args.input_list
    output = args.output
    num_counters = args.num_counters

    parse_csv(input_list, num_counters, output)

if __name__ == '__main__':
    main() 
