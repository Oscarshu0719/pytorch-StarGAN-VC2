# pytorch-StarGAN-VC2

This is a pytorch implementation of the paper: [StarGAN-VC2: Rethinking Conditional Methods for StarGAN-Based Voice Conversion](https://arxiv.org/abs/1907.12279).

**\*\*Under construction\*\***

## Testing environment
- Ubuntu 18.04
- Intel i7-9700KF
- Nvidia RTX 2070 Super
- 32 GB RAM

## Dependencies
- Python 3.6+
- pytorch 1.0
- librosa 
- pyworld 
- tensorboardX
- scikit-learn

## Usage

### Download dataset

#### Download the vcc 2016 dataset to `./data`

``` bash
python download.py 
```

#### Move the corresponding speakers to the training and testing directories

``` bash
bash move_data.sh
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
```

### Convert

```
python main.py --mode test
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