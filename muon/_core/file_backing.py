from os import PathLike
from os.path import abspath
from typing import Optional
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
        super().__init__(None, abspath(filename), filemode)

    def open(
        self,
        filename: Optional[PathLike] = None,
        filemode: Optional[ad.compat.Literal["r", "r+"]] = None,
    ) -> bool:
        if self._file is not None and (
            filename is None
            and filemode is None
            or filename == self.filename
            and filemode == self._filemode
            and self._file.id
        ):
            self._counter += 1
            return False

        if self._file is not None and self._file.id:
            self._file.close()

        if filename is not None:
            self.filename = filename
        if filemode is not None:
            self._filemode = filemode
        if self.filename is None:
            raise ValueError("Cannot open backing file if backing not initialized")
        self._file = h5py.File(self.filename, self._filemode)
        self._counter = 1
        for child in self._children:
            child._file = self._file["mod"][child._mod]
        return True

    def close(self):
        for child in self._children:
            child.close()

    def _close(self):
        if self._counter > 0:
            self._counter -= 1
            if self._counter == 0:
                self._file.close()

    def _to_memory_mode(self):
        for m in self._children:
            m._to_memory_mode()
        self._file.close()
        self._file = None
        self.filename = None

    @property
    def is_open(self) -> bool:
        return (self._file is not None) & bool(self._file.id)

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
        super().__init__(adata, parent.filename, parent._filemode)

    def open(
        self,
        filename: Optional[PathLike] = None,
        filemode: Optional[ad.compat.Literal["r", "r+"]] = None,
    ):
        if not self._parent.open(filename, filemode):
            self._file = self._parent._file["mod"][self._mod]

    def close(self):
        self._parent._close()

    def _to_memory_mode(self):
        self._adata.__X = self._adata.X[()]
        self.close()
        self.filename = None

    @property
    def is_open(self) -> bool:
        return self._parent.is_open
