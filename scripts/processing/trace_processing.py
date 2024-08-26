import argparse
import pandas

def parse_csv(input1, index, output1, output2):
    # Using readlines()
    input_file= open(input1, 'r')
    csv_files = input_file.readlines()

    merged_df = pandas.read_csv('blank.csv')
    initial_read = 0

    for csv in csv_files:
        csv = csv.strip()
        with open(csv, 'r+') as fp:
            lines = fp.readlines()
            fp.seek(0)
            fp.writelines('time,value,,counter,,,,\n')
            fp.writelines(lines[2:])

        # read the csv file
        df1 = pandas.read_csv(csv)
        df1['time'] = df1['time'].astype(float).round(2) #.round({'time':2})
        print(df1)
        # Remove unused columns
        df1.drop(df1.columns[4:], axis=1, inplace=True)

        # Remove unusued intermediate column
        df1.drop(df1.columns[2], axis=1, inplace=True)

        # Get same counter
        df3 = df1.iloc[index-0::2, :]
        counter_name = df3.iloc[index-0][2]
        counter_name = counter_name[:-3]
        df3 = df3.rename(columns={'value': counter_name})
        df3.drop(df3.columns[2], axis=1, inplace=True)

        df4 = df1.iloc[index-1::2, :]
        counter_name = df4.iloc[index-0][2]
        counter_name = counter_name[:-3]
        df4 = df4.rename(columns={'value': counter_name})

        # Remove unusued intermediate column
        df4.drop(df4.columns[2], axis=1, inplace=True)

        df5 = pandas.merge(df3, df4, on = 'time', how='inner')
        df5 = df5.round({'time':2})
        print(list(df5.columns))
        print(df5)

        if initial_read == 0:
            merged_df = pandas.merge(merged_df, df5, on = 'time', how = 'inner')
            initial_read = 1
        else:
#            df5.drop(df5.columns['instruction'], axis=1, inplace=True)
            df5 = df5.drop(columns=['instruction'])
#            merged_df = merged_df.append(df5, ignore_index=True)
#            frames = [merged_df, df5]
#            merged_df = pd.concat(frames)

            merged_df = pandas.merge(merged_df, df5, on = 'time', how = 'inner')
        print(merged_df)


        df3.to_csv(output1, sep=',', index=False)
        df4.to_csv(output2, sep=',', index=False)

        merged_df.to_csv('merge.txt', sep=',', index=False)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-1', type=str, default="",
                        help='csv to read from')
    parser.add_argument('--num-counters', type=int, default='',
                        help='number of perf counters collected')
    parser.add_argument('--output-1', type=str, default='',
                        help='Name of output file')
    parser.add_argument('--output-2', type=str, default='',
                        help='Name of output file')

    args = parser.parse_args()
    input1 = args.input_1
    output1 = args.output_1
    output2 = args.output_2
    num_counters = args.num_counters

    parse_csv(input1, num_counters, output1, output2)

if __name__ == '__main__':
    main() 
