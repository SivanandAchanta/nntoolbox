#!/usr/bin/python3.5

'''
This program takes EHMM lab files as input and writes numeric text features as
output

Inputs:
[1] Unique phones list
[2] EHMM lab directory

Outputs:
[1] Output directory

Note1: This is intended for seq2seq/end2end learning and hence durations are 
not used. 

Author: Sivanand Achanta

Date V0: 03-09-2017

'''

import argparse
import os
import csv


# Create a dictionary for unique phones in the dataset
def uniq_phns(opt):
    '''
    Inputs:
    [1] opt.uniqphns_file: file containing the uniq phones of the language
     
    Outputs:
    [1] phns_dict: dictionary with phones as keys and numeric indices as values
    '''

    with open(opt.uniqphns_file) as f:
        phns = [line[:-2] for line in f]

    phns_dict = {}
    for i,j in enumerate(phns):
        phns_dict[j] = i

    return(phns_dict)


# Read EHMM Label file
def read_ehmmfile(in_file):
    '''
    Inputs:
    [1] in_file: ehmm lab file

    Outputs:
    [1] phone_list: list of phones in the lab file
    '''

    fidr = open(in_file,'r')
    fidr.readline() # remove the first line (#)
    ehmm_obj = csv.reader(fidr, delimiter=' ', )

    phone_list = [col[2] for col in ehmm_obj]

    return(phone_list)


def convert_ph2id(phns_dict, phone_list):
    phone_id = [phns_dict[phn] for phn in phone_list]
    return(phone_id)


# Helper function to process the entire EHMM directory
def process_ehmmdir(phns_dict, opt):

    for f in os.listdir(opt.ehmm_dir):
        fname, ext = os.path.splitext(f)
        if ext == '.lab':
            print('Processing file ' + fname)
            labfile = os.path.join(opt.ehmm_dir, f)
            phone_list = read_ehmmfile(labfile)
            
            # convert phone_list to phone_id (numeric format)
            phone_id = convert_ph2id(phns_dict, phone_list)
         
            # write the list to output file  
            out_file = opt.out_dir + fname + '.tfeat'
            fo = open(out_file, 'w')

            for item in phone_id:
                fo.write("%s\n" % item)            
            
            fo.close()



if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniqphns_file', required=True, help='uniqphns.txt')
    parser.add_argument('--ehmm_dir', required=True, help='/voices/lab/')
    parser.add_argument('--out_dir', required=True, help='../feats/tfeats/')

    opt = parser.parse_args()
    print(opt)

    # prepare the output directories
    try:
        os.makedirs(opt.out_dir)
    except OSError:
        pass

    # make uniqe phones dictionary 
    phns_dict = uniq_phns(opt)
    print(phns_dict)
    print(len(phns_dict))

    # process ehmm dir to extract text feats
    process_ehmmdir(phns_dict, opt)

     
		



