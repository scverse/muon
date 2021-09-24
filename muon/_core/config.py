from mudata._core import config as mudata_config
import logging as log

OPTIONS = {}

_VALID_OPTIONS = {}


class set_options:
    """
    Control muon options.

    MuData options are passed to MuData's set_options.

    Available options:

    - ``display_style``: MuData object representation to use
      in notebooks. Use ``'text'`` (default) for the plain text
      representation, and ``'html'`` for the HTML representation.

    Options can be set in the context:

    >>> with mu.set_options(display_style='html'):
    ...     print("Options are applied here")

    ... or globally:

    >>> mu.set_options(display_style='html')
    """

    def __init__(self, **kwargs):
        self.opts = {}
        self.mudata_opts = {}
        for k, v in kwargs.items():
            if k not in OPTIONS and k not in mudata_config.OPTIONS:
                raise ValueError(f"There is no option '{k}' available")
            if k in _VALID_OPTIONS:
                if not _VALID_OPTIONS[k](v):
                    raise ValueError(f"Value '{v}' for the option '{k}' is invalid.")
            if k in OPTIONS:
                self.opts[k] = OPTIONS[k]
            else:
                # For mudata options, there validity is going to be checked by mudata
                self.mudata_opts[k] = mudata_config.OPTIONS[k]
        mudata_config.set_options(**{k: v for k, v in kwargs.items() if k in mudata_config.OPTIONS})
        if self.opts:
            self._apply(**{k: v for k, v in kwargs.items() if k in OPTIONS})

    def _apply(self, opts):
        OPTIONS.update(opts)

    def __enter__(self):
        log.info("Using custom muon options in the new context...")
        return {**mudata_config.OPTIONS, **OPTIONS}

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.info(f"Returning to the previously defined mudata options: {self.mudata_opts}")
        log.info(f"Returning to the previously defined muon options: {self.opts}")
        mudata_config.set_options(**self.mudata_opts)
        self._apply(self.opts)
