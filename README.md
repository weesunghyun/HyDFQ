# CVPR 2023 paper - Hard Sample Matters a Lot in Zero-Shot Quantization [paper]()

## Requirements

Python >= 3.7.10

Pytorch == 1.8.1

## Reproduce results

### Stage1: Generate data.

take cifar10 as an example:
```
cd data_generate
```
"--save_path_head" in **run_generate_cifar10.sh/run_generate_imagenet.sh** is the path where you want to save your generated data pickle.

You can also call the generator directly. The script defaults to `--model resnet18`, `--batch_size 32`, `--test_batch_size 128`, `--group 1`, `--beta 1.0`, `--gamma 0.0`, and an empty `--save_path_head` (writes next to the script). For example:

```
python generate_data.py --model resnet18 --batch_size 32 --group 1 --beta 1.0 --gamma 0.0 --save_path_head ./generated/
```

Outputs are pickled shards named `<save_path_head>/<model>_refined_gaussian_hardsample_beta<beta>_gamma<gamma>_group<group>.pickle` and `<save_path_head>/<model>_labels_hardsample_beta<beta>_gamma<gamma>_group<group>.pickle`, ready for the Stage2 configuration paths.

```
bash run_generate_cifar10.sh
```


### Stage2: Train the quantized network

```
cd ..
```
1. Modify "qw" and "qa" in cifar10_resnet20.hocon to select desired bit-width.

2. Modify "dataPath" in cifar10_resnet20.hocon to the real dataset path (for construct the test dataloader).

3. Modify "generateDataPath" and "generateLabelPath" in cifar10_resnet20.hocon to the prefixes of the generated pickle shards from Stage1. `main_direct.py` loads four groups named `<generateDataPath>{1..4}.pickle` and `<generateLabelPath>{1..4}.pickle`, so point these fields to the common prefix (ending in `_group`) that comes before the group index.

4. Place the Stage1 synthetic/OOD shards where the paths above resolve. CIFAR variants apply 32×32 random resized crops with horizontal flips, while ImageNet/other datasets apply 224×224 crops and flips when reading these shards.

5. Use the commands in run.sh to train the quantized network. Please note that the model that generates the data and the quantized model should be the same.
