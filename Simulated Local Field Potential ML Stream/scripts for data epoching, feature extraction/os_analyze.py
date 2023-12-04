'''
Jonathan Ramos
11/11/2023
Here, I am just constructing and executing the terminal command to
run the feature extraction script. Sometimes commands can get long when the
arguments are paths and this way, I also minimize the risk of typos in the
terminal command. 
'''
import glob
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='executes specified analyses on epoched, cleaned, curated ephys recording data.')
parser.add_argument('-wn', '--window', action='store_true', help='if true, compute mean pwr and coupling metrics over time freq windows, write to csv')
args = parser.parse_args()

cue0_data = '/Users/jonathanramos/Desktop/CCA_2023/curation_2023/data_curated/CUE0/*.npy'
cue1_data = '/Users/jonathanramos/Desktop/CCA_2023/curation_2023/data_curated/CUE1/*.npy'

if args.window:
    # # CUE0
    # files = glob.glob(cue0_data)
    # if not len(files) == 0:
    #
    #     # parse out set of partial match strings which groups data across regions
    #     partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
    #     for match in partial_match:
    #         HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
    #         os.system(f'python3 window_trials.py {PFC} {HPC}')

    # CUE1
    files = glob.glob(cue1_data)
    if not len(files) == 0:

        # parse out set of partial match strings which groups data across regions
        partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
        print(partial_match)
        sys.exit
        for match in partial_match:
            if not 'ChABC' in match:
                print(match)
                HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
                os.system(f'python3 window_trials.py {PFC} {HPC}')
