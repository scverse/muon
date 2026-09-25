# Install muon

## Stable version

`muon` can be installed [from PyPI](https://pypi.org/project/muon) with `pip`:

```shell
pip install muon
```

## Development version

To use a pre-release version of `muon`, install it from [from the GitHub repository](https://github.com/gtca/muon):

```shell
pip install git+https://github.com/gtca/muon
```

## Troubleshooting

Please see details on installing `scanpy` and its dependencies {doc}`here <scanpy:installation>`.
If there are issues that have not beed described, addressed, or documented, please consider [opening an issue](https://github.com/gtca/muon/issues).

If you encounter the error `Illegal instruction: 4` when installing `muon` on Apple Silicon (e.g. M1 or M2 chips), try [these steps](https://developer.apple.com/metal/tensorflow-plugin/) that were suggested for a similar error when installing TensorFlow.

## Hacking on muon

We use [hatch](https://hatch.pypa.io/) as our build system.
After installing it, you can run `hatch test` or `hatch run docs:build` from the muon project directory.
Happy hacking!
