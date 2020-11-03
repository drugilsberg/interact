[![Build Status](https://travis-ci.org/drugilsberg/interact.svg?branch=master)](https://travis-ci.org/drugilsberg/interact)
[![DOI](https://zenodo.org/badge/167734956.svg)](https://zenodo.org/badge/latestdoi/167734956)
# INtERAcT

The `interact` package provide an implementation of INtERAcT (Interaction networks from vector representation of words) together with a selection of different metrics. Additionally, it exposes various utilities parse data from STRING DB, UniProt and files in gmt format.

INtERAcT is also available as a service on [IBM Cloud](https://ibm.biz/interact-aas). For details check the [paper](https://arxiv.org/abs/1801.03011).

If you use INtERAcT in your research, please consider citing:

```bibtex
@article{manica2019context,
  title={Context-specific interaction networks from vector representation of words},
  author={Manica, Matteo and Mathis, Roland and Cadow, Joris and Mart{\'\i}nez, Mar{\'\i}a Rodr{\'\i}guez},
  journal={Nature Machine Intelligence},
  volume={1},
  number={4},
  pages={181--190},
  year={2019},
  publisher={Nature Publishing Group}
}
```

## Installation

### Setup of the virtual environment

We strongly recommend to work inside a virtual environment (`venv`).

Create the environment:

```sh
python3 -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

### Install dependencies

```sh
pip3 install -r requirements.txt
```

### Module installation

The module can be installed either in editable mode:

```sh
pip3 install -e .
```

Or as a normal package:

```sh
pip3 install .
```

Check the folder `examples` for a quick start on inferring interaction from an embedding using `interact`.
