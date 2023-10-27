import anndata as ad
import networkx as nx
import scanpy as sc
import os
from scglue import *
from itertools import chain
import seaborn as sns
import matplotlib.pyplot as plt
def main():
    print(os.getcwd())
    # glue=models.load_model("glue-modified1014.dill")#glue-modified1012.dill
    glue = models.load_model("/Users/meiqiliu/PycharmProjects/GLUE3_1/evaluation/workflow/scripts/glue/fine-tune/fine-tune.dill")
    # rna = ad.read_h5ad("rna-pp-seq-1014.h5ad")

    rna = ad.read_h5ad("rna-pp2.h5ad")
    atac = ad.read_h5ad("atac-pp2.h5ad")
    guidance = nx.read_graphml("guidance2.graphml.gz")

    models.configure_dataset(
        rna, "NB", use_highly_variable=True,
        use_layer="counts", use_rep="X_pca"
    )
    models.configure_dataset(
        atac, "NB", use_highly_variable=True,
        use_rep="X_lsi"
    )
    # guidance = nx.read_graphml("guidance-seq-2000.graphml.gz")
    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    combined = ad.concat([rna, atac])
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")  # need embedding here
    sc.tl.umap(combined)
    sc.pl.umap(combined, color=["cell_type", "domain"], wspace=0.65)
    dx = models.integration_consistency(
        glue, {"rna": rna, "atac": atac}, guidance_hvf
    )
    print(dx)
    print(sns.lineplot(x="n_meta", y="consistency", data=dx).axhline(y=0.05, c="darkred", ls="--"))
    plt.savefig('line_plot1026.png')






if __name__=="__main__":
    main()