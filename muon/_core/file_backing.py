from pathlib import Path
from os import PathLike
from os.path import abspath
from typing import Optional, Iterator
from collections import defaultdict
from weakref import WeakSet

import anndata as ad
from anndata._core.file_backing import AnnDataFileManager
import h5py


class MuDataFileManager(AnnDataFileManager):
    def __init__(
        self,
        filename: Optional[PathLike] = None,
        filemode: Optional[ad.compat.Literal["r", "r+"]] = None,
    ):
        self._counter = 0
        self._children = WeakSet()
        if filename is not None:
            filename = Path(filename)
        super().__init__(None, filename, filemode)

    def open(
        self,
        filename: Optional[PathLike] = None,
        filemode: Optional[ad.compat.Literal["r", "r+"]] = None,
        add_ref=False,
    ) -> bool:
        if self.is_open and (
            filename is None
            and filemode is None
            or filename == self.filename
            and filemode == self._filemode
        ):
            if add_ref:
                self.counter += 1
            return False

        if self.is_open:
            self._file.close()

        if filename is not None:
            self.filename = filename
        if filemode is not None:
            self._filemode = filemode
        if self.filename is None:
            raise ValueError("Cannot open backing file if backing not initialized")
        self._file = h5py.File(self.filename, self._filemode)
        self._counter = int(add_ref)
        for child in self._children:
            child._set_file()
        return True

    def close(self):
        for child in self._children:
            child.close()

    def _close(self):
        if self._counter > 0:
            self._counter -= 1
        if self._counter == 0 and self.is_open:
            self._file.close()

    def _to_memory_mode(self):
        for m in self._children:
            m._to_memory_mode()
        self._file.close()
        self._file = None
        self.filename = None

    @property
    def is_open(self) -> bool:
        return (self._file is not None) and bool(self._file.id)

    @AnnDataFileManager.filename.setter
    def filename(self, filename: Optional[PathLike]):
        self._filename = None if filename is None else filename


class AnnDataFileManager(ad._core.file_backing.AnnDataFileManager):
    _h5files = {}

    def __init__(
        self,
        adata: ad.AnnData,
        mod: str,
        parent: MuDataFileManager,
    ):
        self._parent = parent
        self._mod = mod
        parent._children.add(self)
        super().__init__(adata)

        if parent.is_open:
            self._set_file()

    def open(
        self,
        filename: Optional[PathLike] = None,
        filemode: Optional[ad.compat.Literal["r", "r+"]] = None,
    ):
        if not self._parent.open(filename, filemode, add_ref=True):
            self._set_file()

    def _set_file(self):
        if self._parent.is_open:
            self._file = self._parent._file["mod"][self._mod]

    @property
    def filename(self) -> Path:
        return self._parent.filename

    @filename.setter
    def filename(self, fname: PathLike):
        pass  # the setter is needed because it's used in ad._core.file_backing.AnnDataFileManager.__init__

    def close(self):
        self._parent._close()

    def _to_memory_mode(self):
        self._adata.__X = self._adata.X[()]
        self.close()
        self.filename = None

    @property
    def is_open(self) -> bool:
        return self._parent.is_open
