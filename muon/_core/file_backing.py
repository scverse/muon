from os import PathLike
from os.path import abspath
from typing import Optional
from collections import defaultdict

import anndata as ad
import h5py


class MuDataFileManager(ad._core.file_backing.AnnDataFileManager):
    _h5files = {}

    def __init__(
        self,
        adata: ad.AnnData,
        mod: str,
        file: Optional[h5py.File] = None,
        filename: Optional[PathLike] = None,
        filemode: Optional[ad.compat.Literal["r", "r+"]] = None,
    ):
        path = abspath(file.filename)

        if file is not None:
            filename = file.filename
            filemode = file.mode

        self._mod = mod

        if path not in self._h5files:
            self._h5files[path] = [file, 0]
        self.__file = self._h5files[path]
        super().__init__(adata, abspath(filename), filemode)

    def open(
        self,
        filename: Optional[PathLike] = None,
        filemode: Optional[ad.compat.Literal["r", "r+"]] = None,
    ):
        if filename is not None:
            self._filename = abspath(filename)
        if filemode is not None:
            self._filemode = filemode
        if self._filename is None:
            raise ValueError("Cannot open backing file if backing not initialized")

        if self.__file[0] is None or not self.__file[0].id:
            self.__file[0] = h5py.File(self._filename, self._filemode)
            self.__file[1] = 1
        else:
            self.__file[1] += 1
        self._file = self.__file[0]['mod'][self._mod]

    def close(self):
        self.__file[1] -= 1
        if self.__file[1] == 0:
            self.__file[0].close()
            self.__file[0] = None

    def _to_memory_mode(self):
        self._adata.__X = self._adata.X[()]
        self.file.close()
        self._filename = None

    @property
    def is_open(self) -> bool:
        if self.__file[0] is None or self.__file[1] == 0:
            return False
        else:
            return bool(self.__file[0].id)
