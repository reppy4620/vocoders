vocoders
===

Experiments for vocoder.

Model list

- 01_hifigan : HiFiGAN
- 02_hifigan_amp : HiFiGAN with AntiAliasActivation
- 03_bigvgan : BigVGAN-base
- 04_vocos : Vocos
- 05_bigvgan_f0 : BigVGAN-base with NSF module
- 06_ms_istft_bigvgan_f0 : BigVGAN-base with NSF module, iSTFT and Learnable PQMF
- 07_ms_bigvgan_f0 : BigVGAN-base with NSF module and Learnable PQMF
- 08_hifigan_f0 : HiFiGAN with NSF module
- 09_ms_hifigan_f0 : HiFiGAN with NSF module and learnalble PQMF
- 10_san_ms_hifigan_f0 : HiFiGAN with NSF module and learnable PQMF and utilizing SAN
- 11_wavenext : WaveNeXt (not stable)
- 12_bigvgan_v2_f0 : BigVGAN-v2(same generator as v1 but the loss and discriminator are different) with NSF module
- 13_bigvsan_v2_f0 : BigVSAN with BigVGAN-v2 setup and NSF module
- 14_bigvsan_f0 : BigVSAN with BigVGAN setup and NSF module
- 15_bigvgan_f0_more : BigVGAN but more channels

# Usage

The directiory for experiment is located in `exp`.

## 0. Install Rye

Please follow the installation instructions [here](https://rye.astral.sh/guide/installation/).

After that, set up the project by running the following command:
```
$ rye sync
```

## 1. Preprocessing (00_preprocess)

1. Split dat to train/valid
2. Extract F0 with Harvest

Default config is located in `src/vocoders/bin/conf/path/dummy.yaml`, so if you run with original dataset, please change the path config file.  
Current system supports single-speaker or universal vocoder, and it is assumed that the audio files to be used are placed in the `wav_dir`.   
Additionally, the audio files are expected to be 1 channel, 16-bit, 24kHz in default settings.

Then run the preprocess script

```sh
$ cd /path/to/vocoders/exp/00_preprocess

$ ./run.sh
```

## 2. Training

Move to the `01_train`directory in prepared experiment directories(e.g. `exp/01_hifigan`) and execute the following command.

```sh
# e.g. : `01_hifigan`
$ cd /path/to/vocoders/exp/01_hifigan/01_train

$ ./run.sh
```

## 3. Synthesis

Move to the `02_syn` directory and execute the following command.  
The default checkpoint file is set as `01_train/out/ckpt/last.ckpt`.  
In the process, validation data is synthesized and evaluated.

```sh
# e.g. : `01_hifigan`
$ cd /path/to/vocoders/exp/01_hifigan/02_syn

$ ./run.sh
```
