
import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
from itertools import chain
from matplotlib import rcParams
import pandas as pd

import os
print(os.getcwd())
# os.chdir('/home/parallels/Downloads')
#
rna = ad.read_h5ad("Chen-2019-RNA.h5ad")
atac = ad.read_h5ad("Chen-2019-ATAC.h5ad")
rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=800, flavor="seurat_v3")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")
sc.pp.neighbors(rna, metric="cosine")
sc.tl.umap(rna)
scglue.data.lsi(atac, n_components=100, n_iter=15)
print(rna,atac)
#
# #
# sc.pp.subsample(rna, n_obs=1000, random_state=42)
# atac=atac[pd.Series(atac.obs_names).sample(1000),pd.Series(atac.var_names).sample(1000)].copy()
#
# #
# scglue.data.get_gene_annotation(
#     rna, gtf="gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
#     gtf_by="gene_name"
# )
# #
# split = atac.var_names.str.split(r"[:-]")
# atac.var["chrom"] = split.map(lambda x: x[0])
# atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
# atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)