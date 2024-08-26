import argparse
import pandas
import numpy as np

def main():
   
    with open('blank.csv','w') as file:
        file.write(str('time'))
        file.write('\n')
        for i in np.arange(0.01, 10000, 0.01):
            file.write(str(round(i,2)))
            file.write('\n')
    
if __name__ == '__main__':
    main() 
