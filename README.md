# tension
The `tension` package is a Python package for building and FORCE training chaotic recurrent neural networks. `tension` is built to integrate seamlessly with TensorFlow/Keras, a widely used software package for training artificial neural networks. 

## Code and documentation
Installation guide, contributing guide, and API reference are available on the [documentation website](https://zhenruiliao.github.io/tension/index.html)

Example notebooks for reproducing the experiments in the associated software report can also be found on the documentation website and in the `examples` folder. These notebooks can be run either in Google Colab or in a local jupyter notebook.

Documentation for TensorFlow/Keras, on which `tension` is based and interoperable with, can be found at the [TensorFlow guide](https://www.tensorflow.org/guide).

## Dependencies
`tension` depends on Python >= 3.7, `tensorflow>=2.5`, `numpy` and `matplotlib`.

## Installation
### Local installation
We recommend installing `tension` into a conda environment
```
conda create -n tension python=3.7
```
followed by
```
conda activate tension
```

#### From GitHub
Clone this repo using
```
git clone https://github.com/zhenruiliao/tension.git
```
Change into the `tension` directory and install using `pip`
```
cd tension/
pip install -e .
```

#### From PyPI
```
pip install tension
```

### Google Colab
To quickly get started with `tension`, the package can also be installed in Google Colab using the following commands
```
!git clone https://github.com/zhenruiliao/tension.git tension
!pip install -e tension
```
The runtime must be restarted for the package to become importable

## Contributing 
Bug reports, feature requests, and pull requests are welcome and encouraged! Use the Issues and Pull requests tabs to open new issues or pull requests. Always be kind and respectful. 

## License
`tension` is provided under the [MIT License](./LICENSE). TensorFlow is provided under the [Apache 2.0 license](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).

## Acknowledgments

Based on work by David Sussillo and Larry Abbott
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/

With thanks to James Priestley for the package name.
