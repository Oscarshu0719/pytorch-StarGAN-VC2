import argparse
import os
import shlex, subprocess
from urllib.request import urlretrieve
from zipfile import ZipFile

def unzip(zip_filepath, dest_dir='./data'):
    with ZipFile(zip_filepath) as zf:
        zf.extractall(dest_dir)
    print(f"{zip_filepath} has been extracted to {dest_dir}.")

def download_dataset(dataset='VCC2016'):
    if dataset == 'VCC2016':
        data_link = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
        data_files = ['vcc2016_training.zip', 'evaluation_all.zip']
    else:
        data_link = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/"
        data_files = ['vcc2018_database_training.zip', 'vcc2018_database_evaluation.zip']

    exit_flag = False
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f'{data_file} exists.')
            exit_flag = True

    if exit_flag:
        return

    train_set = f'{data_link}/{data_files[0]}'
    eval_set = f'{data_link}/{data_files[1]}'

    train_comm = f'wget {train_set}'
    eval_comm = f'wget {eval_set}'

    train_comm = shlex.split(train_comm)
    eval_comm = shlex.split(eval_comm)

    print(f'Start downloading dataset {dataset}...')
    
    subprocess.run(train_comm)
    subprocess.run(eval_comm)

    unzip(data_files[0])
    unzip(data_files[1])
    
    print(f'Finish downloading dataset {dataset}...')

def create_dirs(train_dir: str='./data/spk', test_dir: str='./data/spk_test'):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download VCC datasets.')

    dataset_default = 'VCC2016'
    train_dir = './data/spk'
    test_dir = './data/spk_test'

    parser.add_argument('--dataset', type=str, default=dataset_default, choices=['VCC2016', 'VCC2018'], 
        help='Available datasets: VCC2016 and VCC2018 (Default: VCC2018).')
    parser.add_argument('--train_dir', type=str, default=train_dir, help='Directory of train set.')
    parser.add_argument('--test_dir', type=str, default=test_dir, help='Directory of test set.')

    argv = parser.parse_args()

    create_dirs(train_dir, test_dir)

    download_dataset(dataset=argv.dataset)
