<div align="right">
  Language:
    🇨🇳
  <a title="English" href="./README.en.md">🇺🇸</a>
  <!-- <a title="Russian" href="../ru/README.md">🇷🇺</a> -->
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/X3D"><img align="center" src="./imgs/X3D.png"></a></div>

<p align="center">
  «X3D» reproduces the video classification model proposed in the paper <a title="" href="https://arxiv.org/abs/2004.04730">X3D: Expanding Architectures for Efficient Video Recognition </a>
  <br>
  <br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

Its `CodeBase` comes from [ZJCV/Non-local](https://github.com/ZJCV/Non-local)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

## Background

The paper authors analyze the development of previous video understanding models in detail, propose `6` key variables, search for better models through progressive testing, and finally obtain `6` `X3D` models at different variable scales.

## Install

Install the dependencies required for running via `requirements.txt`

```
$ pip install -r requirements.txt
```

Data processing additionally requires [denseflow](https://github.com/open-mmlab/denseflow); installation scripts can be found in [innerlee/setup](https://github.com/innerlee/setup).

## Usage

First set the `GPU` and the current location

```
$ export CUDA_VISIBLE_DEVICES=1
$ export PYTHONPATH=.
```

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Acknowledgements

* [ facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)
* [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2)

```
@misc{feichtenhofer2020x3d,
      title={X3D: Expanding Architectures for Efficient Video Recognition}, 
      author={Christoph Feichtenhofer},
      year={2020},
      eprint={2004.04730},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contributing

Contributions from anyone are welcome! Open an [issue](https://github.com/ZJCV/X3D/issues) or submit a pull request.

Notes:

* For `GIT` commits, please follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/) specification
* For semantic versioning, please follow the [Semantic Versioning 2.0.0](https://semver.org) specification
* For `README` files, please follow the [standard-readme](https://github.com/RichardLitt/standard-readme) specification

## License

[Apache License 2.0](LICENSE) © 2020 zjykzj
