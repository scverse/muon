# Contributing


This document describes details about contributing to `muon`.

The main entry point for a contribution is an issue. Please use issues to discuss the change you wish to make or the funcionality you want to add to `muon`. For a more in-depth discussion you can also use [discussions](https://github.com/PMBio/muon/discussions) or contact `muon` authors or maintainers via other communication methods such as email.

## Issues

Please consider opening an issue if you've encountered a bug, a performance issue, a documentation issue or have a feature request in mind. For convenience, we provide issue templates that you are very welcome to use.

When creating an issue about a problem that you've encountered (e.g. an error), please include the minimal amount of source code to reproduce it. When including tracebacks, please paste the full traceback text.

## Pull requests

The code that is suggested to be merged into `muon` is expected to follow reasonable Python code styleguides such as the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Below there are a few ideas that may help to improve the code quality.

- Format the code with [black](https://github.com/psf/black).
- Make sure debugging code (e.g. `pdb.set_trace()`) is removed as well as respective dependencies (`import pdb`).
- Use tools like `pylint` and `flake8` to check proposed code changes.
- Make sure documentation is changed to reflect the changes. That includes docstrings as well as external files such as the ones in `docs/` or respective `README.md` files.
- Consider increasing the version number in `setup.py`. Please stick to [semantic versioning](https://semver.org/).
- Pull requests can be merged when the LGTM (_looks good to me_) has been received from reviewers, probably after a few rounds of reviews.

