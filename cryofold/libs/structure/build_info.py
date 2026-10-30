"""Hard-coded values to use when building and analyzing sidechain structures.

Bond lengths and angles are programmatically parsed from AMBER forcefields.
'p' and 'i' for 'torsion-vals' stand for predicted and inferred, respectively.
"""

NUM_ANGLES = 12
NUM_BB_TORSION_ANGLES = 3
NUM_BB_OTHER_ANGLES = 3
NUM_SC_ANGLES = NUM_ANGLES - (NUM_BB_OTHER_ANGLES + NUM_BB_TORSION_ANGLES)
SC_ANGLES_START_POS = NUM_BB_OTHER_ANGLES + NUM_BB_TORSION_ANGLES

NUM_COORDS_PER_RES = 14
PRODY_CA_DIST = 4.1

ATOM_COUNTS = [1, 5, 4, 8, 8, 7, 7, 11, 14, 12, 8, 9, 11, 10, 9, 6, 7, 6, 8, 8, 9, 4]


CHAIN_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


CLS_RESIDUE_DICT = {
    0: "BG_",
    1: "ALA",
    2: "GLY",
    3: "ILE",
    4: "LEU",
    5: "PRO",
    6: "VAL",
    7: "PHE",
    8: "TRP",
    9: "TYR",
    10: "ASP",
    11: "GLU",
    12: "ARG",
    13: "HIS",
    14: "LYS",
    15: "SER",
    16: "THR",
    17: "CYS",
    18: "MET",
    19: "ASN",
    20: "GLN",
    21: "UNK",
}

RESIDUE_CLS_DICT = {
    "BG_": 0, 
    "ALA": 1,
    "GLY": 2,
    "ILE": 3,
    "LEU": 4,
    "PRO": 5,
    "VAL": 6,
    "PHE": 7,
    "TRP": 8,
    "TYR": 9,
    "ASP": 10,
    "GLU": 11,
    "ARG": 12,
    "HIS": 13,
    "LYS": 14,
    "SER": 15,
    "THR": 16,
    "CYS": 17,
    "MET": 18,
    "ASN": 19,
    "GLN": 20,
    "UNK": 21,
    "CA": 21,
    "BMA": 21,
    "HOH": 21,
    "HEM": 21,
    "MAN": 21,
    "Y5P": 21,
    "P5P": 21,
    "BRU": 21,
    "M7M": 21,
    "MN": 21,
    "RAP": 21,
    "AGS": 21,
    "ATP": 21,
    "GTP": 21,
    "CTP": 21,
    "DTP": 21,
    "SEP": 21,
    "ADP": 21,
    "GDP": 21,
    "NAG": 21,
    "SY9": 21,
    "IVM": 21,
    "MG": 21,
    "ZN": 21,
    "DA": 21,
    "DC": 21,
    "DG": 21,
    "DT": 21,
    "A": 21,
    "C": 21,
    "G": 21,
    "U": 21,
}

RESIDUE_CLASS_DICT = {
    "ALA": 1,
    "GLY": 2,
    "ILE": 3,
    "LEU": 4,
    "PRO": 5,
    "VAL": 6,
    "PHE": 7,
    "TRP": 8,
    "TYR": 9,
    "ASP": 10,
    "GLU": 11,
    "ARG": 12,
    "HIS": 13,
    "LYS": 14,
    "SER": 15,
    "THR": 16,
    "CYS": 17,
    "MET": 18,
    "ASN": 19,
    "GLN": 20,
}

REPLACEMENTS = {
        "UNK": "ALA",
        "ASX": "ASP",
        "GLX": "GLU",
        "MEQ": "GLN",
        "CSO": "CYS",
        "HIP": "HIS",
        "HSD": "HIS",
        "HSE": "HIS",
        "HSP": "HIS",
        "HSO": "HIS",
        "MSE": "MET",
        "FME": "MET",
        "SEC": "CYS",
        "SEP": "SER",
        "TPO": "THR",
        "PTR": "TYR",
        "XLE": "LEU",
        "4FB": "PRO",
        "MLY": "LYS"  # N-dimethyl-lysine
    }

RESIDUE_NAMES = RESIDUE_CLASS_DICT.keys() 

RESTYPE_NAME_TO_ATOM14_NAMES = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'BG_': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],

}

SC_BUILD_INFO = {
    'BG_': {
        'angles-names': ['N-CA-CB'],
        'angles-types': ['N -CX-CT'],
        'angles-vals': [1.9146261894377796],
        'atom-names': ['CB'],
        'bonds-names': ['CA-CB'],
        'bonds-types': ['CX-CT'],
        'bonds-vals': [1.526],
        'torsion-names': ['C-N-CA-CB'],
        'torsion-types': ['C -N -CX-CT'],
        'torsion-vals': ['p'],
        'diangle-names': ['N_C_CA_CB'],
        'diangle-vals': [2.1412746461017633],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'ALA': {
        'angles-names': ['N-CA-CB'],
        'angles-types': ['N -CX-CT'],
        'angles-vals': [1.9146261894377796],
        'atom-names': ['CB'],
        'bonds-names': ['CA-CB'],
        'bonds-types': ['CX-CT'],
        'bonds-vals': [1.526],
        'torsion-names': ['C-N-CA-CB'],
        'torsion-types': ['C -N -CX-CT'],
        'torsion-vals': ['p'],
        'diangle-names': ['N_C_CA_CB'],
        'diangle-vals': [2.1412746461017633],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'ARG': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-NE', 'CD-NE-CZ', 'NE-CZ-NH1',
            'NE-CZ-NH2'
        ],
        'angles-types': [
            'N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-N2', 'C8-N2-CA', 'N2-CA-N2',
            'N2-CA-N2'
        ],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.9408061282176945,
            2.150245638457014, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-NE', 'NE-CZ', 'CZ-NH1', 'CZ-NH2'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-N2', 'N2-CA', 'CA-N2', 'CA-N2'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.463, 1.34, 1.34, 1.34],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-NE', 'CG-CD-NE-CZ',
            'CD-NE-CZ-NH1', 'CD-NE-CZ-NH2'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-N2', 'C8-C8-N2-CA',
            'C8-N2-CA-N2', 'C8-N2-CA-N2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p', 'p', 'i'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD','CB_CG_CD_NE','CG_CD_NE_CZ','CD_NE_CZ_NH1','CD_NE_CZ_NH2'],
        'diangle-vals0': [122.76,-65.2,-179.2,-179.3,-178.7, 0.0, 180.0],
        'diangle-vals': [ 2.14256619, -1.13795467, -3.12763002, -3.12937535, -3.11890337, 0.,  3.14159265],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'ASN': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-ND2'],
        'angles-types': ['N -CX-2C', 'CX-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.101376419401173, 2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'ND2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-ND2'],
        'bonds-types': ['CX-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.522, 1.229, 1.335],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-ND2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-C ', 'CX-2C-C -O ', 'CX-2C-C -N '],
        'torsion-vals': ['p', 'p', 'p', 'i'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_OD1','CA_CB_CG_ND2'],
        'diangle-vals0': [123.2254,-65.5,-58.3,180-58.3],
        'diangle-vals': [ 2.15068895, -1.14319066, -1.01752695,  2.1240657 ],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'ASP': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-OD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.0420352248333655, 2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'OD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-OD2'],
        'bonds-types': ['CX-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.522, 1.25, 1.25],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-OD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-CO', 'CX-2C-CO-O2', 'CX-2C-CO-O2'],
        'torsion-vals': ['p', 'p', 'p', 'i'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_OD1','CA_CB_CG_OD2'],
        'diangle-vals0': [122.82,-66.4,-46.7,180-46.7],
        'diangle-vals': [ 2.14361339, -1.15889862, -0.81506876,  2.32652389],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'CYS': {
        'angles-names': ['N-CA-CB', 'CA-CB-SG'],
        'angles-types': ['N -CX-2C', 'CX-2C-SH'],
        'angles-vals': [1.9146261894377796, 1.8954275676658419],
        'atom-names': ['CB', 'SG'],
        'bonds-names': ['CA-CB', 'CB-SG'],
        'bonds-types': ['CX-2C', '2C-SH'],
        'bonds-vals': [1.526, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-SG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-SH'],
        'torsion-vals': ['p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_SG'],
        'diangle-vals0': [122.5037,-62.2],
        'diangle-vals': [ 2.13809291, -1.08559479],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'GLN': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-NE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.101376419401173,
            2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-NE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.526, 1.522, 1.229, 1.335],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-NE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-C ', '2C-2C-C -O ', '2C-2C-C -N '
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD','CB_CG_CD_OE1','CB_CG_CD_NE2'],
        'diangle-vals0': [122.8134,-60.2,-69.6,-50.5,180-50.5],
        'diangle-vals': [ 2.1434982 , -1.05068821, -1.21474916, -0.88139127,  2.26020138],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'GLU': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-OE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.0420352248333655,
            2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-OE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.526, 1.522, 1.25, 1.25],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-OE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-CO', '2C-2C-CO-O2', '2C-2C-CO-O2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD','CB_CG_CD_OE1','CB_CG_CD_OE2'],
        'diangle-vals0': [122.8702,-63.8,-179.8,-6.2,180-6.2],
        'diangle-vals': [ 2.14448954, -1.11352006, -3.138102  , -0.10821041,  3.03338224],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'GLY': {
        'angles-names': [],
        'angles-types': [],
        'angles-vals': [],
        'atom-names': [],
        'bonds-names': [],
        'bonds-types': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': [],
        'diangle-names': [],
        'diangle-vals': [],
        'diangle-mask': [],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'HIS': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-ND1', 'CG-ND1-CE1', 'ND1-CE1-NE2', 'CE1-NE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CC', 'CT-CC-NA', 'CC-NA-CR', 'NA-CR-NB', 'CR-NB-CV'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9739673840055867, 2.0943951023931953,
            1.8849555921538759, 1.8849555921538759, 1.8849555921538759
        ],
        'atom-names': ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-ND1', 'ND1-CE1', 'CE1-NE2', 'NE2-CD2'],
        'bonds-types': ['CX-CT', 'CT-CC', 'CC-NA', 'NA-CR', 'CR-NB', 'NB-CV'],
        'bonds-vals': [1.526, 1.504, 1.385, 1.343, 1.335, 1.394],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-ND1', 'CB-CG-ND1-CE1', 'CG-ND1-CE1-NE2',
            'ND1-CE1-NE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CC', 'CX-CT-CC-NA', 'CT-CC-NA-CR', 'CC-NA-CR-NB',
            'NA-CR-NB-CV'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_ND1','CA_CB_CG_CD2','CB_CG_ND1_CE1','CB_CG_CD2_NE2'],
        'diangle-vals0': [122.6711,-63.2,-75.7,180-75.7,180.0,180.0],
        'diangle-vals': [ 2.14101459, -1.10304809, -1.32121424,  1.82037841,  3.14159265,3.14159265],
        'order': [0,1,2,3,4,5,6,8,9,7,10,11,12,13],
    },
    'ILE': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CB-CG1-CD1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-2C', '3C-2C-CT', 'CX-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG1', 'CD1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CG1-CD1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-2C', '2C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'CA-CB-CG1-CD1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-2C', 'CX-3C-2C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG1','N_CA_CB_CG2','CA_CB_CG1_CD1'],
        'diangle-vals0': [123.2347,59.7,-61.6,169.8],
        'diangle-vals': [ 2.15085127,  1.04196156, -1.07512282,  2.96356907],
        'order': [0,1,2,3,4,5,7,6,8,9,10,11,12,13],
    },
    'LEU': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CB-CG-CD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-3C', '2C-3C-CT', '2C-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD1', 'CG-CD2'],
        'bonds-types': ['CX-2C', '2C-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CA-CB-CG-CD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-3C', 'CX-2C-3C-CT', 'CX-2C-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD1','CA_CB_CG_CD2'],
        'diangle-vals0': [122.4948,-60.1,174.9,66.7],
        'diangle-vals': [ 2.13793758, -1.04894288,  3.05258086,  1.16413461],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'LYS': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-CE', 'CD-CE-NZ'],
        'angles-types': ['N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-C8', 'C8-C8-N3'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791,
            1.9408061282176945
        ],
        'atom-names': ['CB', 'CG', 'CD', 'CE', 'NZ'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-CE', 'CE-NZ'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-C8', 'C8-N3'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526, 1.471],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-CE', 'CG-CD-CE-NZ'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-C8', 'C8-C8-C8-N3'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD','CB_CG_CD_CE','CG_CD_CE_NZ'],
        'diangle-vals0': [122.76,-64.5,-178.1,-179.6,179.6],
        'diangle-vals': [ 2.14256619, -1.12573737, -3.1084314 , -3.13461134,  3.13461134],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'MET': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-SD', 'CG-SD-CE'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-S ', '2C-S -CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 2.0018926520374962, 1.726130630222392
        ],
        'atom-names': ['CB', 'CG', 'SD', 'CE'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-SD', 'SD-CE'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-S ', 'S -CT'],
        'bonds-vals': [1.526, 1.526, 1.81, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-SD', 'CB-CG-SD-CE'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-S ', '2C-2C-S -CT'],
        'torsion-vals': ['p', 'p', 'p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_SD','CB_CG_SD_CE'],
        'diangle-vals0': [122.6733,-64.5,-179.6,70.1],
        'diangle-vals': [ 2.14105299, -1.12573737, -3.13461134,  1.22347581],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'PHE': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-CE2',
            'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CA',
            'CA-CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': ['CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA'],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.4, 1.4, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-CA',
            'CA-CA-CA-CA', 'CA-CA-CA-CA'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0, 0.0]
        ,
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD1','CA_CB_CG_CD2','CB_CG_CD1_CE1','CB_CG_CD2_CE2','CG_CD1_CE1_CZ'],
        'diangle-vals0': [122.6054,-64.7,93.3,93.3-180,180.0,180.0,0.0],
        'diangle-vals': [ 2.13986791, -1.12922803,  1.62839219, -1.51320046,  3.14159265,3.14159265,  0.        ],
        'order': [0,1,2,3,4,5,6,8,10,9,7,11,12,13],
    },
    'PRO': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD'],
        'angles-types': ['N -CX-CT', 'CX-CT-CT', 'CT-CT-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG', 'CD'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD'],
        'bonds-types': ['CX-CT', 'CT-CT', 'CT-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD'],
        'torsion-types': ['C -N -CX-CT', 'N -CX-CT-CT', 'CX-CT-CT-CT'],
        'torsion-vals': ['p', 'p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD'],
        'diangle-vals0': [115.2975,29.6,-34.8],
        'diangle-vals': [ 2.01232099,  0.51661746, -0.60737458],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'SER': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG'],
        'angles-types': ['N -CX-2C', 'CX-2C-OH'],
        'angles-vals': [1.9146261894377796, 1.911135530933791],
        'atom-names': ['CB', 'OG'],
        'bonds-names': ['CA-CB', 'CB-OG'],
        'bonds-types': ['CX-2C', '2C-OH'],
        'bonds-vals': [1.526, 1.41],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-OH'],
        'torsion-vals': ['p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_OG'],
        'diangle-vals0': [122.6618,-63.3],
        'diangle-vals': [ 2.14085228, -1.10479342],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'THR': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-OH', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'OG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-OG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-OH', '3C-CT'],
        'bonds-vals': [1.526, 1.41, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-OH', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_OG1','N_CA_CB_CG2'],
        'diangle-vals0': [123.0953,60.0,-60.3],
        'diangle-vals': [ 2.14841828,  1.04719755, -1.05243354],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'TRP': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-NE1', 'CD1-NE1-CE2',
            'NE1-CE2-CZ2', 'CE2-CZ2-CH2', 'CZ2-CH2-CZ3', 'CH2-CZ3-CE3', 'CZ3-CE3-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-C*', 'CT-C*-CW', 'C*-CW-NA', 'CW-NA-CN', 'NA-CN-CA',
            'CN-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CB'
        ],
        'angles-vals': [
            1.9146261894377796, 2.0176006153054447, 2.181661564992912, 1.8971728969178363,
            1.9477874452256716, 2.3177972466484698, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': [
            'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2'
        ],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-NE1', 'NE1-CE2', 'CE2-CZ2', 'CZ2-CH2',
            'CH2-CZ3', 'CZ3-CE3', 'CE3-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-C*', 'C*-CW', 'CW-NA', 'NA-CN', 'CN-CA', 'CA-CA', 'CA-CA',
            'CA-CA', 'CA-CB'
        ],
        'bonds-vals': [1.526, 1.495, 1.352, 1.381, 1.38, 1.4, 1.4, 1.4, 1.4, 1.404],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-NE1', 'CG-CD1-NE1-CE2',
            'CD1-NE1-CE2-CZ2', 'NE1-CE2-CZ2-CH2', 'CE2-CZ2-CH2-CZ3', 'CZ2-CH2-CZ3-CE3',
            'CH2-CZ3-CE3-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-C*', 'CX-CT-C*-CW', 'CT-C*-CW-NA', 'C*-CW-NA-CN',
            'CW-NA-CN-CA', 'NA-CN-CA-CA', 'CN-CA-CA-CA', 'CA-CA-CA-CA', 'CA-CA-CA-CB'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 3.141592653589793,
            0.0, 0.0, 0.0
        ],
        'diangle-names': [
            'N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD1','CA_CB_CG_CD2',
            'CB_CG_CD1_NE1','CB_CG_CD2_CE2','CB_CG_CD2_CE3',
            'CG_CD2_CE2_CZ2','CG_CD2_CE3_CZ3','CD2_CE2_CZ2_CH2'],
        'diangle-vals0': [122.6112,-66.4,96.3,96.3-180, 180.0,180.0,0.0,180.0,180.0,0.0],
        'diangle-vals': [ 2.13996914, -1.15889862,  1.68075207, -1.46084058,  3.14159265, 3.14159265,  0.        ,  3.14159265,  3.14159265,  0.        ],
        'order': [0,1,2,3,4,5,6,8,9,11,13,12,10,7],
    },
    'TYR': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-OH',
            'CE1-CZ-CE2', 'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-C ', 'CA-C -OH',
            'CA-C -CA', 'C -CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'OH', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-OH', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-C ', 'C -OH', 'C -CA', 'CA-CA'
        ],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.409, 1.364, 1.409, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-OH', 'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-C ',
            'CA-CA-C -OH', 'CA-CA-C -CA', 'CA-C -CA-CA'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 0.0, 0.0
        ],
        'diangle-names': [
            'N_C_CA_CB','N_CA_CB_CG','CA_CB_CG_CD1','CA_CB_CG_CD2',
            'CB_CG_CD1_CE1','CB_CG_CD2_CE2','CG_CD1_CE1_CZ','CD1_CE1_CZ_OH'],
        'diangle-vals0': [122.6023,-64.3,93.1,93.1-180, 180.0,180.0,0.0,180.0],
        'diangle-vals': [2.13981381, -1.12224671,  1.62490153, -1.51669112,  3.14159265, 3.14159265,  0.        ,  3.14159265],
        'order': [0,1,2,3,4,5,6,8,10,11,9,7,12,13],
    },
    'VAL': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-CT', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p'],
        'diangle-names': ['N_C_CA_CB','N_CA_CB_CG1','N_CA_CB_CG2'],
        'diangle-vals0': [123.2347,177.2,-63.3],
        'diangle-vals': [2.15085127,  3.09272343, -1.10479342],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
    'UNK': {
        'angles-names': ['N-CA-CB'],
        'angles-types': ['N -CX-CT'],
        'angles-vals': [1.9146261894377796],
        'atom-names': ['CB'],
        'bonds-names': ['CA-CB'],
        'bonds-types': ['CX-CT'],
        'bonds-vals': [1.526],
        'torsion-names': ['C-N-CA-CB'],
        'torsion-types': ['C -N -CX-CT'],
        'torsion-vals': ['p'],
        'diangle-names': ['N_C_CA_CB'],
        'diangle-vals': [2.1412746461017633],
        'order': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    },
}

BB_BUILD_INFO = {
    "BONDLENS": {
        'n-ca': 1.442,
        'ca-c': 1.498,
        'c-n': 1.379,
        'c-o': 1.229,  # From parm10.dat
        'c-oh': 1.364
    },  # From parm10.dat, for OXT
    # For placing oxygens
    "BONDANGS": {
        'OMEGA':  3.14159265,
        'N-CA-C': 1.91986218,
        'CA-C-N': 2.03580428,
        'C-N-CA': 2.11851932,
        'CA-C-O': 2.09439510, # Approximated to be 2pi / 3; parm10.dat says 2.0350539
    },  # Equal to 'ca-c-o', for OXT
    "BONDTORSIONS": {
        'n-ca-c-n': -0.785398163
    },  # A simple approximation, not meant to be exact.
    "BB_ANG": {
        'OMEGA':  3.14159265,
        'N-CA-C': 1.91986218,
        'CA-C-N': 2.03580428,
        'C-N-CA': 2.11851932,
        'CA-C-O': 2.09439510,
    }#[3.14159265, 1.91986218, 2.03580428, 2.11851932]
}
