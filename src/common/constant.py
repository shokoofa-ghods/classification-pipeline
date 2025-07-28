INITIAL_LABELS = {
    'PLAX': 0,
    'PSAX-ves': 1,
    'PSAX-base': 2,
    'PSAX-mid': 3,
    'PSAX-apical': 4,
    'Apical-2ch': 5,
    'Apical-3ch': 6,
    'Apical-5ch': 7,
    'Apical-4ch': 8,
    'Suprasternal': 9,
    'Subcostal': 10
}

COMBINED_VIEW_MAP = {
    0 : 'plax',          # PLAX
    1: 'psax-ves',     # PSAX-ves
    2: 'psax-sub',     # PSAX-base
    3: 'psax-sub',     # PSAX-mid
    4: 'psax-sub',     # PSAX-apical
    5: 'apical-2ch',   # Apical-2ch
    6: 'apical-3ch',   # Apical-3ch
    7: 'apical-4&5ch',   # Apical-5ch
    8: 'apical-4&5ch',   # Apical-4ch
    9: 'suprasternal',   # Suprasternal   
    10: 'subcostal'      # Subcostal
}

VIEW_MAP = ['plax', 'psax-ves', 'psax-sub', 'apical-2ch', 'apical-3ch', 'apical-4&5ch', 'suprasternal', 'subcostal']
