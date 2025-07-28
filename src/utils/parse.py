"""
parser module
"""

import re
from typing import Any

TRUE_REGEX      = re.compile(r'\s*[tT][rR][uU][eE]\s*\Z')
FALSE_REGEX     = re.compile(r'\s*[fF][aA][lL][sS][eE]\s*\Z')
HEX_INT_REGEX   = re.compile(r'\s*[+-]?0[xX](?:_?[0-9a-fA-F])+\s*\Z')
OCT_INT_REGEX   = re.compile(r'\s*[+-]?0[oO](?:_?[0-7])+\s*\Z')
BIN_INT_REGEX   = re.compile(r'\s*[+-]?0[bB](?:_?[01])+\s*\Z')
DEC_INT_REGEX   = re.compile(r'\s*[+-]?[0-9](?:_?[0-9])*\s*\Z')
FLOAT_REGEX_1   = re.compile(r'\s*[+-]?[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?\s*\Z')
FLOAT_REGEX_2   = re.compile(r'\s*[+-]?\.[0-9](?:_?[0-9])*\s*\Z')
FLOAT_REGEX_3   = re.compile(r'\s*[+-]?[0-9](?:_?[0-9])*[eE][+-]?[0-9](?:_?[0-9])*\s*\Z')
FLOAT_REGEX_4   = re.compile(r'\s*[+-]?[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?[eE][+-]?'
                             r'[0-9](?:_?[0-9])*\s*\Z')
FLOAT_REGEX_5   = re.compile(r'\s*[+-]?\.[0-9](?:_?[0-9])*[eE][+-]?[0-9](?:_?[0-9])*\s*\Z')

def parse(val:str) -> tuple[Any, Any]:
    """
    Parse a string to the closest literal
    Note: ~50% faster than matching by try except tree
    
    Args:
        val (str): The value to parse
    Returns:
        Tuple of the parsed value and type.
        The value will be:
        True if val = "True" (case insensitive),
        False if val = "False" (case insensitive),
        int(val) if val is a valid integer,
        float(val) if val is a valid float; or
        val if no other patterns are matched
    Raises:
        TypeError if val is None
    """

    if TRUE_REGEX.match(val) is not None:
        return True, bool
    if FALSE_REGEX.match(val) is not None:
        return False, bool
    if BIN_INT_REGEX.match(val) is not None:
        return int(val, 2), int
    if OCT_INT_REGEX.match(val) is not None:
        return int(val, 8), int
    if DEC_INT_REGEX.match(val) is not None:
        return int(val, 10), int
    if HEX_INT_REGEX.match(val) is not None:
        return int(val, 16), int
    if FLOAT_REGEX_1.match(val) is not None:
        return float(val), float
    if FLOAT_REGEX_2.match(val) is not None:
        return float(val), float
    if FLOAT_REGEX_3.match(val) is not None:
        return float(val), float
    if FLOAT_REGEX_4.match(val) is not None:
        return float(val), float
    if FLOAT_REGEX_5.match(val) is not None:
        return float(val), float
    return val, str
