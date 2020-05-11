# pytorch-StarGAN-VC2

This is a PyTorch implementation of the paper: [StarGAN-VC2: Rethinking Conditional Methods for StarGAN-Based Voice Conversion](https://arxiv.org/abs/1907.12279). 

Support [VCC2016](http://www.vc-challenge.org/vcc2016/index.html) and [VCC2018](http://www.vc-challenge.org/vcc2018/index.html) two datasets.

**\*\*Under construction\*\***

## Dependencies
- Python 3.6+
- pytorch 1.0+
- scikit-learn
- librosa 
- pyworld 
- tensorboardX

## Test environment
- Ubuntu 18.04 LTS
- Intel i7-9700KF
- Nvidia RTX 2070 Super
- 32 GB RAM
- python 3.7.7
- pytorch 1.4.0
- cudnn 7.6.5
- scikit-learn 0.22.1
- librosa 0.6.1
- pyworld 0.2.8
- tensorboardx 2.0

## Usage

### Download dataset

#### Download the vcc 2016 dataset to `./data`.

``` bash
python download.py 
```

#### Move the corresponding speakers to the training and testing directories.

```
bash move_data.sh [dataset] [speakers_list]

dataset:
    VCC2016 or VCC2018 (case insensitive).
    e.g. Use VCC2016 dataset.
        VCC2016
speakers_list
    List of training and testing speakers.
    e.g. Train and test four speakers SF1, SF2, TM1, and TM2.
        SF1, SF2, TM1, TM2

e.g. Use dataset VCC2016. Train and test four speakers SF1, SF2, TM1, and TM2.
    bash move_data.sh VCC2016 SF1 SF2 TM1 TM2
```

1. **Training set**: According to the paper, move direcotories SF1, SF2, TM1, TM2 to `./data/spk`.
2. **Testing set**: According to the paper, move direcotories SF1, SF2, TM1, TM2 to `./data/spk_test`.

Directory structure: 

```
data
├── spk 
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── spk_test
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── vcc2016_training (vcc 2016 training set)
│   ├── ...
├── evaluation_all (vcc 2016 evaluation set, we use it as testing set)
│   ├── ...
```

### Preprocess

Extract WORLD features (mcep, f0, ap) from each speech clip.

```
python preprocess.py
```

### Train

```
python main.py

Notice:
    Details of other arguments are in `main.py`.
```

### Convert

```
python main.py --mode convert [--src_speaker src_speaker] [--trg_speaker trg_speaker_list] [--test_iters test_iters]

src_speaker:
    Source of speaker (quotation marks are neccessary).
    e.g. Source speaker is VCC2SM1.
        "VCC2SM1"

trg_speaker_list:
    List of target of speakers (quotation marks are neccessary).
    e.g. Target speakers are VCC2SM1 and VCC2SF1.
        "['VCC2SM1', 'VCC2SF1']"

e.g. Convert from speaker VCC2SM1 to speakers VCC2SM1 and VCC2SM1 at step 100000.
    python main.py --mode convert --src_speaker "VCC2SM1" --trg_speaker "['VCC2SM1', 'VCC2SM1']" --test_iters 100000
    
Notice:
    Details of other arguments are in `main.py`.
```

## References

### Code

[pytorch-StarGAN-VC](https://github.com/hujinsen/pytorch-StarGAN-VC) by [hujinshen](https://github.com/hujinsen).

[StarGAN-Voice-Conversion-2](https://github.com/SamuelBroughton/StarGAN-Voice-Conversion-2) by [SamuelBroughton](https://github.com/SamuelBroughton).

### Paper

[StarGAN-VC2: Rethinking Conditional Methods for StarGAN-Based Voice Conversion](https://arxiv.org/abs/1907.12279)

[StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks](https://arxiv.org/abs/1806.02169)

## License

MIT License.
