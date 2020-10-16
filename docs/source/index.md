# Get to know muon

## Multimodal data containers

`muon` introduces multimodal data containers (`MuData` class) allowing Python users to work with increasigly complex datasets efficiently and to build new workflows and computational tools around it.

```
MuData object with n_obs × n_vars = 10110 × 110101
 2 modalities
  atac: 10110 x 100001
  rna: 10110 x 10100
```

## Multi-omics methods

`muon` brings multi-omics methods availability to a whole new level: state-of-the-art methods for multi-omics data integration are just a function call away.

```py
mu.tl.mofa(mdata)
```

## Methods crafted for omics

`muon` features methods for specific omics such as ATAC-seq and CITE-seq making it an extendable solution and allowing for prospective growth in an open-source environment.

```py
ac.pp.tfidf(mdata.mod['atac'])
```

Contents
--------
* [API Reference](api/index.md)
