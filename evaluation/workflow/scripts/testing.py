import bindome as bd
import anndata as ad
import scanpy as sc
import scglue
import pandas as pd
from itertools import chain
keys=['rna', 'atac']
import networkx as nx
import scglue
#define adatas
rna1 = ad.read_h5ad("rna-pp2.h5ad")

atac1 = ad.read_h5ad("atac-pp2.h5ad")
atac1=atac1[pd.Series(atac1.obs_names).sample(1000),pd.Series(atac1.var_names).sample(1000)].copy()
#
scglue.models.configure_dataset(
    rna1, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_pca"
)

scglue.models.configure_dataset(
    atac1, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)
guidance = nx.read_graphml("guidance2.graphml.gz")

adatas={"rna": rna1, "atac": atac1}
pretrain_init_kws={}
pretrain_init_kws.update({"shared_batches": False})


pretrain = scglue.models.scglue.SCGLUEModel(adatas,sorted(guidance.nodes),**pretrain_init_kws)
modalities=pretrain.modalities

from scglue.models.data import AnnDataset, ArrayDataset, DataLoader, GraphDataset


anndata = AnnDataset(
            [adatas[key] for key in keys],
            [modalities[key] for key in keys],
            mode="train"
        )

data_loader = DataLoader( #three elements: x, xrep, _
            anndata, batch_size=128,
            shuffle=False, drop_last=False
        )

print(pretrain.net)

print(anndata[2000])

for i in range(len(anndata[0])):
    data=anndata[0][i]
    print("{}th data===========".format(i), len(data))
    print(data)


for i,d in enumerate(anndata.extracted_data):
    print("{}th element===========".format(i),len(d))
    print(d[0].shape)

