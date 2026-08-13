import pandas as pd

def parse_mutation(df):
    '''
    Parse single amino acid substitutions (i.e. A45F), and adds additional features

    Adds:
        wt_aa
        mut_aa
        position
        protein_length
        relative_position
    '''

    df = df.copy()

    df['wt_aa'] = df['mut_type'].str.extract(r"^([A-Z])")

    df['mut_aa'] = df['mut_type'].str.extract(r".*[0-9]([A-Z])$")

    df['position'] = df['mut_type'].str.extract(r"^[A-Z]([0-9]+)[A-Z]$", expand = False).astype(int)

    df['protein_length'] = df['aa_seq'].str.len()
    
    df["relative_position"] = (
        df["position"] / df["protein_length"]
    )

    return df