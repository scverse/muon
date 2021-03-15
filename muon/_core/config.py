import logging as log

OPTIONS = {
    "display_style": "text",
    "display_html_expand": 0b010,
}

_VALID_OPTIONS = {
    "display_style": lambda x: x in ("text", "html"),
    "display_html_expand": lambda x: isinstance(x, int) and len(bin(x or 0b111)) == 5,
}


class set_options:
    """
    Control muon options.

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
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(f"There is no option '{k}' available")
            if k in _VALID_OPTIONS:
                if not _VALID_OPTIONS[k](v):
                    raise ValueError(f"Value '{v}' for the option '{k}' is invalid.")
            self.opts[k] = OPTIONS[k]
        self._apply(kwargs)

    def _apply(self, opts):
        OPTIONS.update(opts)

    def __enter__(self):
        log.info("Using custom muon options in the new context...")
        return OPTIONS

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.info(f"Returning to the previously defined options: {self.opts}")
        self._apply(self.opts)
