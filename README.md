# Semantic Image Segmentation of Eucalyptus Trees in Panoramic Imagery: Approaches of Deep Learning for Forest Management
## Introduction

This work assesses the performance of novel deep-learning methods for eucalyptus tree segmentation in panoramic RGB ground-level images. We applied four models: the FCN, GCNet, ANN, and PointRend, using a challenging dataset composed of eucalyptus trees with variation in distances between trunks, curvature, sizes, and trunks of different diameters. The four semantic segmentation methods were trained and evaluated in five-fold cross-validation. A quantitative-qualitative analysis is presented, along with a discussion about the advantages and limitations of each CNN applied.

The master branch works with **PyTorch 1.3+**.

## Dataset 
The datasets of the models used in this work are available at [https://drive.google.com/drive/folders/19-xnaSfwRppgkPlfLazYeBMXI3cxY_sJ?usp=sharing](https://drive.google.com/drive/folders/19-xnaSfwRppgkPlfLazYeBMXI3cxY_sJ?usp=sharing) and should be placed in the `dataset` folder.

## Checkpoints
The checkpoints of the models used in this work are available at [https://drive.google.com/drive/folders/19-xnaSfwRppgkPlfLazYeBMXI3cxY_sJ?usp=sharing](https://drive.google.com/drive/folders/19-xnaSfwRppgkPlfLazYeBMXI3cxY_sJ?usp=sharing) and should be placed in the `checkpoints` folder.
## License

This project is released under the [MIT license](LICENSE).

## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [train.md](docs/train.md) and [inference.md](docs/inference.md) for the basic usage of MMSegmentation.
There are also tutorials for [customizing dataset](docs/tutorials/customize_datasets.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing modules](docs/tutorials/customize_models.md), and [customizing runtime](docs/tutorials/customize_runtime.md).
We also provide many [training tricks](docs/tutorials/training_tricks.md).

## Citation

If you find this project useful in your research, please consider cite:

```latex
@unpublished{carvalho2024semantic,
  author = {Mário de Araújo Carvalho and José Marcato Junior and Amaury Antônio de Castro Junior and Celso Soares Costa and Pedro Alberto Pereira Zamboni and José Augusto Correa Martins and Lucas Prado Osco and Michelle Taís Garcia Furuya and Felipe David Georges Gomes and Ana Paula Marques Ramos and Henrique Lopes Siqueira and Diogo Nunes Gonçalves and Jonathan Li and Wesley Nunes Gonçalves},
  title = {Semantic Image Segmentation of Eucalyptus Trees in Panoramic Imagery: Approaches of Deep Learning for Forest Management},
  note = {Submitted to Ecological Informatics, under review},
  year = {2024},
  url = {https://github.com/MarioCarvalhoBr/segmentation-eucalyptus-trees-in-panoramic-images}
}

```

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement and Reference
This project is based on the project - [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods. Documentation of MMSegmentation is available at [https://mmsegmentation.readthedocs.io](https://mmsegmentation.readthedocs.io).
