"""Multiple non-destructive Unicode representations; no country-specific whitelist."""
import re
import unicodedata

def light(text:str)->str:
    """Preserve letters, combining marks and numbers from every script."""
    text=unicodedata.normalize('NFC',text).lower()
    return ' '.join(''.join(c if unicodedata.category(c)[0] in 'LMN' else ' ' for c in text).split())

def compatible(text:str)->str:
    """NFKC and casefold are optional representations, never raw replacements."""
    return light(unicodedata.normalize('NFKC',text).casefold().replace('&',' and '))

def latin_accent_fold(text:str)->str:
    """Remove diacritics only from Latin letters, preserving Indic vowel signs."""
    out=[];latin=False
    for ch in unicodedata.normalize('NFD',text):
        if unicodedata.category(ch).startswith('M'):
            if not latin:out.append(ch)
        else:
            latin='LATIN' in unicodedata.name(ch,'');out.append(ch)
    return light(unicodedata.normalize('NFC',''.join(out)))
