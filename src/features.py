import pandas as pd

def parse_mutations(df):
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

## Create dictionaries of biochemical properties

## From Kyte & Doolittle (1982)
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5,
    "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}
## From International Union of Pure and Applied Chemistry (IUPAC)
AA_WEIGHT = {
    'A': 89.09,   # Alanine
    'R': 174.20,  # Arginine
    'N': 132.12,  # Asparagine
    'D': 133.10,  # Aspartic acid
    'C': 121.16,  # Cysteine
    'E': 147.13,  # Glutamic acid
    'Q': 146.15,  # Glutamine
    'G': 75.07,   # Glycine
    'H': 155.15,  # Histidine
    'I': 131.17,  # Isoleucine
    'L': 131.17,  # Leucine
    'K': 146.19,  # Lysine
    'M': 149.21,  # Methionine
    'F': 165.19,  # Phenylalanine
    'P': 115.13,  # Proline
    'S': 105.09,  # Serine
    'T': 119.12,  # Threonine
    'W': 204.23,  # Tryptophan
    'Y': 181.19,  # Tyrosine
    'V': 117.15   # Valine
}

## Nelson, D. L., & Cox, M. M. (2021). Lehninger principles of biochemistry (8th ed.). W. H. Freeman.
AA_CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 
    'E': -1, 'Q': 0, 'G': 0, 'H': 0, 'I': 0, 
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

## Nelson, D. L., & Cox, M. M. (2021). Lehninger principles of biochemistry (8th ed.). W. H. Freeman.
AA_POLARITY = {
    'A': 'nonpolar', 'R': 'basic', 'N': 'polar', 'D': 'acidic', 'C': 'polar',
    'E': 'acidic', 'Q': 'polar', 'G': 'nonpolar', 'H': 'basic', 'I': 'nonpolar',
    'L': 'nonpolar', 'K': 'basic', 'M': 'nonpolar', 'F': 'nonpolar', 'P': 'nonpolar',
    'S': 'polar', 'T': 'polar', 'W': 'nonpolar', 'Y': 'polar', 'V': 'nonpolar'
}

## Nelson, D. L., & Cox, M. M. (2021). Lehninger principles of biochemistry (8th ed.). W. H. Freeman.
AA_AROMATIC = {
    'A': False, 'R': False, 'N': False, 'D': False, 'C': False,
    'E': False, 'Q': False, 'G': False, 'H': True, 'I': False,
    'L': False, 'K': False, 'M': False, 'F': True, 'P': False,
    'S': False, 'T': False, 'W': True, 'Y': True, 'V': False
}

def get_biochemical_features(df):
    '''
    Take a df containing wt and mutant amino acids and provided biochemical data for both including
    delta between mut - wt if applicable.

        adds:
            hydrophobicity
            molecular weight 
            charge
            polarity
            aromatic
    '''
    df = df.copy()

    ## Get Hydrophobicity
    df['wt_hydrophobicity'] = df['wt_aa'].map(HYDROPHOBICITY)
    df['mut_hydrophobicity'] = df['mut_aa'].map(HYDROPHOBICITY)
    ## Change in Hydrophobicity
    df["delta_hydrophobicity"] = (df["mut_hydrophobicity"] - df["wt_hydrophobicity"])

    ## Get Molecular weight 
    df['wt_mw'] = df['wt_aa'].map(AA_WEIGHT)
    df['mut_mw'] = df['mut_aa'].map(AA_WEIGHT)
    ## Change in Molecular weight 
    df["delta_mw"] = (df["mut_mw"] - df["wt_mw"])

    ## Get Charge
    df['wt_charge'] = df['wt_aa'].map(AA_CHARGE)
    df['mut_charge'] = df['mut_aa'].map(AA_CHARGE)
    ## Change in Charge
    df["delta_charge"] = (df["mut_charge"] - df["wt_charge"])

    ## Get Polarity
    df['wt_polarity'] = df['wt_aa'].map(AA_POLARITY)
    df['mut_polarity'] = df['mut_aa'].map(AA_POLARITY)

    ## Get Aromatic
    df['wt_aromatic'] = df['wt_aa'].map(AA_AROMATIC)
    df['mut_aromatic'] = df['mut_aa'].map(AA_AROMATIC)

    return df