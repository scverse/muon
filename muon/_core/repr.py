#
# Utility functions for MuData._repr_html_()
#

from typing import Tuple, Iterable
import pandas as pd


def maybe_module_class(obj, sep=".", builtins=False) -> Tuple[str, str]:
    m, cl = "", obj.__class__.__name__
    try:
        m += obj.__class__.__module__
        if m == "builtins" and not builtins:
            m = ""
        else:
            m += sep
    except:
        m += ""
    return (m, cl)


def format_values(x):
    s = ""
    if not isinstance(x, Iterable):
        s += f"{x}"
    elif isinstance(x, pd.DataFrame):
        s += "DataFrame ({} x {})".format(*x.shape)
    elif hasattr(x, "keys") and hasattr(x, "values") and not hasattr(x, "shape"):
        ks = ",".join(x.keys())
        if "," not in ks:  # only 1 element
            vs = ",".join(map(format_values, x.values()))
            s += ks + ": " + vs
        else:
            s += ks
    elif isinstance(x, str):
        s += x
    else:
        x = x[: min(100, len(x))]
        if isinstance(x[0], float):
            s += ",".join([f"{i:.2f}" for i in x])
        else:
            s += ",".join([f"{i}" for i in x])
        s = s[:50]
        while s[-1] != ",":
            s = s[:-1]
        s += "..."
    return s


def block_matrix(data, attr, name):
    obj = getattr(data, attr)
    s = ""
    s += "<div class='title title-attr'>{}</div><span class='hl-dim'>.{}</span>".format(name, attr)
    s += "<div>"
    s += """
            <span class='hl-types'>{}</span> <span>&nbsp;&nbsp;&nbsp;<span class='hl-import'>{}</span>{}</span>
         """.format(
        obj.dtype, *maybe_module_class(obj)
    )
    s += "</table></div>"
    return s


def details_block_table(data, attr, name, expand=0, dims=True, square=False):
    obj = getattr(data, attr)
    s = ""
    # DataFrame
    if isinstance(obj, pd.DataFrame):
        s += "<details{}>".format(" open" if expand else "")
        s += "<summary><div class='title title-attr'>{}</div><span class='hl-dim'>.{}</span><span class='hl-size'>{} element{}</span></summary>".format(
            name, attr, obj.shape[1], "s" if obj.shape[1] != 1 else ""
        )
        if obj.shape[1] > 0:
            s += "<div><table>"
            s += "\n".join(
                [
                    """<tr>
                                <td class='col-index'>{}</td>  <td class='hl-types'>{}</td>  <td class='hl-values'>{}</td>
                            </tr>""".format(
                        attr_key, obj[attr_key].dtype, format_values(obj[attr_key])
                    )
                    for attr_key in obj.columns
                ]
            )
            s += "</table></div>"
        else:
            s += f"<span class='hl-empty'>No {name.lower()}</span>"
        s += "</details>"
    # Dict-like object
    elif hasattr(obj, "keys") and hasattr(obj, "values") and name != "Miscellaneous":
        s += "<details{}>".format(" open" if expand else "")
        s += "<summary><div class='title title-attr'>{}</div><span class='hl-dim'>.{}</span><span class='hl-size'>{} element{}</span></summary>".format(
            name, attr, len(obj), "s" if len(obj) != 1 else ""
        )
        if len(obj) > 0:
            s += "<div><table>"
            if square:  # e.g. distance matrices in .obsp
                s += "\n".join(
                    [
                        """<tr>
                                       <td class='col-index'>{}</td>  <td class='hl-types'>{}</td>  <td><span class='hl-import'>{}</span>{}</td>
                                   </tr>""".format(
                            attr_key, obj[attr_key].dtype, *maybe_module_class(obj[attr_key])
                        )
                        for attr_key in obj.keys()
                    ]
                )
            else:  # e.g. embeddings in .obsm
                s += "\n".join(
                    [
                        """<tr>
                                       <td class='col-index'>{}</td>  <td class='hl-types'>{}</td>  <td><span class='hl-import'>{}</span>{}</td>  <td class='hl-dims'>{}</td>
                                   </tr>""".format(
                            attr_key,
                            obj[attr_key].dtype,
                            *maybe_module_class(obj[attr_key]),
                            f"{obj[attr_key].shape[1]} dims"
                            if len(obj[attr_key].shape) > 1 and dims
                            else "",
                        )
                        for attr_key in obj.keys()
                    ]
                )

            s += "</table></div>"
        else:
            s += f"<span class='hl-empty'>No {name.lower()}</span>"
        s += "</details>"
    elif hasattr(obj, "file"):  # HDF5 dataset
        s += "<details{}>".format(" open" if expand else "")
        s += "<summary><div class='title title-attr'>{}</div><span class='hl-dim'>.{}</span><span class='hl-size'>{} elements</span></summary>".format(
            name, attr, len(obj)
        )
        s += "<div><table>"
        s += """<tr>
                <td class='hl-types'>{}</td>  <td><span class='hl-import'>{}</span>{}</td>
                </tr>""".format(
            obj.dtype, *maybe_module_class(obj)
        )
        s += "</table></div>"
        s += "</details>"
    else:  # Unstructured
        s += "<details{}>".format(" open" if expand else "")
        s += "<summary><div class='title title-attr'>{}</div><span class='hl-dim'>.{}</span><span class='hl-size'>{} elements</span></summary>".format(
            name, attr, len(obj)
        )
        if len(obj) > 0:
            s += "<div><table>"
            s += "\n".join(
                [
                    """<tr>
                                <td class='col-index'>{}</td>  <td><span class='hl-import'>{}</span>{}</td>  <td class='hl-dims'>{} element{}</td>  <td class='hl-values'>{}</td>
                            </tr>""".format(
                        attr_key,
                        *maybe_module_class(obj[attr_key]),
                        len(obj[attr_key]),
                        "s" if len(obj[attr_key]) != 1 else "",
                        format_values(obj[attr_key]),
                    )
                    for attr_key in obj.keys()
                ]
            )
            s += "</table></div>"
        else:
            s += f"<span class='hl-empty'>No {name.lower()}</span>"
        s += "</details>"
    return s


MUDATA_CSS = """<style>
.hl-dim, .hl-size, .hl-values, .hl-types, .hl-dims {
  color: #777777;
}
.hl-dim::before, .hl-size::before {
  content: "\\00a0\\00a0\\00a0";
}
.hl-values {
  font-family: monospace;
}
.hl-file {
  background-color: #EEEEEE;
  border-radius: .5rem;
  padding: .2rem .4rem;
  color: #555555;
}
.hl-empty {
  color: #999999;
}
.hl-import {
  color: #777777;
}
.block-mod {
  display: block;
  margin: 0 2rem;
}
.title {
  display: inline-block;
  font-weight: 600;
  color: #555555;
}
.title-mod {
  font-size: 1.2rem;
  color: #04b374;
  padding: 0 .5rem;
}
.title-attr {
  font-size: 1.0rem;
  padding-top: .2rem;
}
summary {
  cursor: pointer;
  list-style: none;
}
summary::-webkit-details-marker {
  display: none;
}
details > summary::before {
  content: '\u2295';
}
details[open] > summary::before {
  content: '\u2296';
}
table tr {
  background-color: transparent !important;
}
table tr:hover {
  background-color: #04b37433 !important;
}
.col-index {
  text-align: left !important;
}
.summary-mod {
  margin-left: -2rem;
}
.summary-mod:hover {
  background-color: #04b37411;
}
.summary-mod::before {
  color: #04b374;
  content: '\u25cf';
}
details[open] > .summary-mod::before {
  content: '\u25cb';
}
</style>"""
