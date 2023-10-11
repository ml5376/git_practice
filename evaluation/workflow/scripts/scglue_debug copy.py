#try use filtered data (rna,atac)
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
# rna = ad.read_h5ad("Chen-2019-RNA.h5ad")
# atac = ad.read_h5ad("Chen-2019-ATAC.h5ad")
# rna.layers["counts"] = rna.X.copy()
# sc.pp.highly_variable_genes(rna, n_top_genes=800, flavor="seurat_v3")
# sc.pp.normalize_total(rna)
# sc.pp.log1p(rna)
# sc.pp.scale(rna)
# sc.tl.pca(rna, n_comps=100, svd_solver="auto")
# sc.pp.neighbors(rna, metric="cosine")
# sc.tl.umap(rna)
# scglue.data.lsi(atac, n_components=100, n_iter=15)
#
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

# guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)# change name to prior
# #scglue.graph.check_graph(guidance, [rna, atac])
#
# rna.write("rna-pp2.h5ad", compression="gzip")
# atac.write("atac-pp2.h5ad", compression="gzip")
# nx.write_graphml(guidance, "guidance2.graphml.gz")

# rna = ad.read_h5ad("rna-pp2.h5ad")
# atac = ad.read_h5ad("atac-pp2.h5ad")
# atac=atac[pd.Series(atac.obs_names).sample(1000),pd.Series(atac.var_names).sample(1000)].copy()


# guidance = nx.read_graphml("guidance2.graphml.gz")
# print('atac.var',atac.var)

rna = ad.read_h5ad("rna-pp-seq.h5ad")
atac = ad.read_h5ad("atac-pp-seq.h5ad")
guidance = nx.read_graphml("guidance-seq.graphml.gz")
# rna = ad.read_h5ad("rna-pp2.h5ad")
# atac = ad.read_h5ad("atac-pp2.h5ad")
# guidance = nx.read_graphml("guidance2.graphml.gz")
scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_pca"
)

scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)
guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance_hvf,
    fit_kws={"directory": "glue"}
)

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

print('rna-embedding',rna.obsm["X_glue"].shape)
print('atac-embedding',atac.obsm["X_glue"].shape)
print(glue)

glue.save("glue-modified.dill")
