import pandas as pd
import logging

def exclude_items(df1,d1):
    for c in df1:
        if c in d1:
            # info(c)
            shape0=df1.shape
            df1=df1.loc[~(df1[c].isin(d1[c])),:]
            if shape0!=df1.shape:
                logging.info(f"{c}:{df1[c].nunique()}")
    return df1

def read_pre_processed(
    p,
    rename: bool=True,
    excludes: dict={}, # directory
    clean: bool=False,
    )->pd.DataFrame:
    """
    Standardise the columns of the pre-processed data.
    
    Parameters:
        p (str|pd.DataFrame): the path to the input table or a dataframe.
        
    """
    ## read the table
    if isinstance(p,str):
        df1=pd.read_csv(p,index_col=0,
                       sep='\t' if p.endswith('.tsv') else ',',
                       )
    elif isinstance(p,pd.DataFrame):
        df1=p.copy()
    else:
        raise ValueError(p)
        
    ## cell id per pair
    if not 'cell id per pair' in df1:
        df1=(df1
             .assign(
                **{
                    'cell id per pair':lambda df: range(len(df)),
                  }             
            )
            )
        logging.info('column added: `cell id per pair`, to be used to map to the other pre-processed data.')
    
    ## URL
    if 'URL' in df1:
        df1['URL']=df1['URL'].apply(lambda x: str(x).zfill(9))
        
    ## filter
    df1=exclude_items(df1,excludes)
    if len(df1)==0:
        if isinstance(p,str):
            logging.info(f"excluded: {p}")
        return 
    
    ## rename columns
    if rename:
        from modules.tools.ids import to_renamed
        df1=to_renamed(
            df1,
            clean=clean,
            )
    return df1

def to_image_separatedby_channel(
    source_bfconvert,
    path_raw_image,
    channel,
    path_output,
    test=False,
    ):
    com=f"bash {source_bfconvert} -nogroup -overwrite -channel {channel} {path_raw_image} {path_output}"
    response=subprocess.call(com,
                   shell=True)
    if test:
        info(com)
    # brk
    assert response==0 and exists(path_output)
    return path_output

## io file
### multiple seq fasta
def read_fasta(
    fap: str,
    key_type: str='id',
    duplicates: bool=False,
    ) -> dict:
    """Read fasta

    Args:
        fap (str): path
        key_type (str, optional): key type. Defaults to 'id'.
        duplicates (bool, optional): duplicates present. Defaults to False.

    Returns:
        dict: data.

    Notes:
        1. If `duplicates` key_type is set to `description` instead of `id`.
    """
    from Bio import SeqIO,SeqRecord,Seq
    if (not duplicates) or key_type=='id':
        try:
            id2seq=SeqIO.to_dict(SeqIO.parse(fap,format='fasta'))
            id2seq={k:str(id2seq[k].seq) for k in id2seq}
            return id2seq
        except:
            duplicates=True
    if duplicates or key_type=='description':
        id2seq={}
        for seq_record in SeqIO.parse(fap, "fasta"):
            id2seq[getattr(seq_record,key_type)]=str(seq_record.seq)
        return id2seq
