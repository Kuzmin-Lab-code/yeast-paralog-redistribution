## attributes
import pandas as pd
    
def to_construct_label(
    statuses=[], # e.g. 'query','partner'
    symbols=[],
    sep: str=' ',
    show_wt=True, # show WT background gene  
    fmt=True,
    ) -> str:
    """
    Create the common label for the constructs.
    
    Parameters: 
        x: row of the table.
        sep: separator between genes.
        
    Returns
        str: common label
    """
    def get_query(s,fmt): 
        return (s.capitalize() if fmt else s.upper())+'-GFP' 
    def get_partner(s,fmt,status):         
        if status=='DELTA':
            s=f"${s.lower()}$" if fmt else s.upper()
            suffix='$\Delta$' if fmt else f'-{status}'
        elif status=='WT':
            if show_wt:
                s=f"${s.upper()}$"  if fmt else s.upper()
                suffix="" if fmt else f'-{status}'
            else:
                s="wild-type"
                suffix=""
        return s+suffix
    assert len(set(statuses) - set(('WT','GFP','DELTA')))==0, statuses
    if   (statuses[0],statuses[1]) == ('GFP','WT'):
        return get_query(symbols[0],fmt=fmt)+sep+get_partner(symbols[1],fmt=fmt,status=statuses[1])
    elif (statuses[0],statuses[1]) == ('GFP','DELTA'):
        return get_query(symbols[0],fmt=fmt)+sep+get_partner(symbols[1],fmt=fmt,status=statuses[1])
    elif (statuses[0],statuses[1]) == ('WT','GFP'):
        return get_query(symbols[1],fmt=fmt)+sep+get_partner(symbols[0],fmt=fmt,status=statuses[0])
    elif (statuses[0],statuses[1]) == ('DELTA','GFP'):
        return get_query(symbols[1],fmt=fmt)+sep+get_partner(symbols[0],fmt=fmt,status=statuses[0])
    else:
        logging.error(f"{symbols} {statuses}")

def to_renamed(
    df0: pd.DataFrame,
    clean=False,
    ) -> pd.DataFrame:
    """
    Rename the columns of the table containing the paired raw data.
    
    Steps:
        1. sort the genes in a pairs.
        2. rename the statuses of the genes.

    Parameters:
        df0 (pd.DataFrame): input table.
        # metadata (dict): metadata.
        
    Returns:
        df0 (pd.DataFrame): output table.
        
    Notes:
        suffixes: gene1 and gene2 are for sorted pairs.
        suffixes: query and partner are for unsorted pairs.
    """
    if 'URL' in df0:
        df0['URL']=df0['URL'].apply(lambda x: str(x).zfill(9))
    if (not 'image id' in df0) and ('replicate' in df0) and ('URL' in df0):       
        df0['image id']=df0.apply(lambda x: f"{x['replicate']}:{x['URL']}",axis=1)
    
    if not 'gene symbol query' in df0:
        if 'GFP' in df0:
            df0['gene symbol query']=df0['GFP'].copy()            
        elif 'label'in df0:
            df0['gene symbol query']=df0['label'].str.split('-',expand=True)[0]
            # assert all(df0['gene symbol query']==df0['GFP']), 'column `gene symbol query` != `GFP`'
        else:
            pass
    if not 'gene symbol partner' in df0:
        if 'label'in df0:
            df0['gene symbol partner']=(df0['label'].str.split('-',expand=True)[1]).str.split(' ',expand=True)[1]
            # assert all(df0['gene symbol query']==df0['GFP']), 'column `gene symbol query` != `GFP`'
        else:
            pass
    if not 'status query' in df0:
        if 'label' in df0:
            df0['status query']=(df0['label'].str.split('-',expand=True)[1]).str.split(' ',expand=True)[0]
    if not 'status partner' in df0: 
        if 'label' in df0:
            df0['status partner']=df0['label'].str.split('-',expand=True)[2]            
        elif 'natMX4' in df0:
            df0['status partner']=df0['natMX4'].apply(lambda x: 'WT' if pd.isnull(x) else 'DELTA')
            if 'status gene1' in df0 and 'status gene2' in df0:
                ## validate that the mappings is correct
                assert ((df0['status gene1']=='DELTA') | (df0['status gene2']=='DELTA')).sum() == (df0['status partner']=='DELTA').sum() 
        else:
            pass
    if (not 'construct label' in df0) and ('status query' in df0) and ('status partner' in df0):
        # mainly for plotting and counting. 
        ## 'construct label' is formatted 'label'
        df0['construct label']=df0.apply(lambda x: to_construct_label(
                                        symbols=[x['gene symbol query'],x['gene symbol partner']],
                                        statuses=[x['status query'],x['status partner']],
                                        fmt=True,
                                        ),
                                        axis=1)
    if (not 'label common' in df0) and ('status query' in df0) and ('status partner' in df0):
        ### general label common between pairs
        df0['label common']=df0.apply(lambda x: to_construct_label(
                                        symbols=['paralog1','paralog2'] if x['gene symbol query']==x['pairs'].split('-')[0] else ['paralog2','paralog1'],
                                        statuses=[x['status query'],x['status partner']],
                                        fmt=False,
                                        ),
                                        axis=1)
    if clean:
        df0=df0.drop(['GFP','natMX4'],axis=1)
    return df0