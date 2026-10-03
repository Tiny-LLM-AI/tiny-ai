#!/usr/bin/env python3
"""Download Leipzig's Vietnamese-only 100K release and prepare a reproducible corpus."""
import hashlib
import json
import re
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'vie_news_2020_100K'
URL = f'https://downloads.wortschatz-leipzig.de/corpora/{NAME}.tar.gz'
RAW = ROOT / 'train-data' / 'raw' / f'{NAME}.tar.gz'
OUT = ROOT / 'train-data' / 'vi-foundation.txt'
ARCHIVE_SHA256 = '7ce0b3643899cc7f22596369e99ded474751396b3bc698002a3a70dc454ba9a4'
# The source is monolingual VI. These additional heuristics reject obvious noise;
# they are not a proof that every proper noun or quotation is Vietnamese.
DIACRITICS = re.compile('[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', re.I)
VI_WORDS = set('và là của có trong được một những các không người cho với đã này về khi từ đến ở cũng theo để nhưng trên sẽ như đó tôi bạn chúng ta họ đang'.split())

def clean(text):
    text = unicodedata.normalize('NFC', ' '.join(text.split()))
    if not 40 <= len(text) <= 600 or re.search(r'https?://|www\.|@|<|>', text):
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters or any('LATIN' not in unicodedata.name(c, '') for c in letters):
        return None
    words = re.findall(r'[^\W\d_]+', text.lower())
    if len(DIACRITICS.findall(text)) / len(letters) < 0.08:
        return None
    if sum(w in VI_WORDS for w in words) < 2:
        return None
    return text

def main():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        tmp = RAW.with_suffix('.part')
        request = urllib.request.Request(URL, headers={'User-Agent': 'LLM-mini educational corpus downloader'})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        tmp.replace(RAW)
    if hashlib.sha256(RAW.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from the verified Leipzig release; existing corpus left unchanged")
    lines, seen, source_ids = [], set(), []
    count = 0
    with tarfile.open(RAW, 'r:gz') as archive:
        # Read only the known member; never extract arbitrary archive paths.
        with archive.extractfile(f'{NAME}/{NAME}-sentences.txt') as source:
            for raw in source:
                count += 1
                sid, sentence = raw.decode('utf-8').rstrip('\r\n').split('\t', 1)
                text = clean(sentence)
                if text and text not in seen:
                    seen.add(text)
                    lines.append(text)
                    source_ids.append(int(sid))
    if count != 100000 or len(lines) < 10000:
        raise ValueError(f'Unexpected corpus: {count} raw / {len(lines)} accepted')
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    metadata = {
        'id': NAME, 'language': 'vi', 'source': URL,
        'sourcePage': 'https://corpora.uni-leipzig.de/en?corpusId=vie_news_2020',
        'terms': 'https://wortschatz.uni-leipzig.de/en/usage',
        'licenseVerification': 'Official terms page blocked by bot challenge during download; archive has no license file. No license grant inferred.',
        'rawSentences': count, 'acceptedSentences': len(lines),
        'bytes': OUT.stat().st_size,
        'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
        'archiveSha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
        'preprocessing': 'NFC, whitespace, 40..600 chars, reject URLs/email/HTML/non-Latin letters, >=8% VI diacritics, >=2 VI common words, exact deduplication',
        'languageCaveat': 'Vietnamese source plus heuristic filtering; names/loanwords may remain. Not a certified language detector.',
    }
    metadata_dir = OUT.parent / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / OUT.with_suffix('.meta.json').name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    (metadata_dir / OUT.with_suffix('.source-ids.json').name).write_text(json.dumps(source_ids) + '\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
