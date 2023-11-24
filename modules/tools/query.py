"""For querying e.g. subset of the data."""
import pandas as pd
import roux.lib.dfs as rd

def get_wt_protein_abundance(
    df01,
    to_gene_ids=None,
    ):
    """
    Notes:
        to_gene_ids=read_table(metadata['ids']['genes']).rd.to_dict(['gene symbol','gene id'])
    """
    return (
        df01
        .log.query(expr="`status partner`=='WT'")
        .rd.assert_no_dups(subset='gene symbol query')
        .assign(
            **{'gene id':lambda df: df['gene symbol query'].map(to_gene_ids)}
        )
        .rd.assert_dense(subset='gene id')
        )