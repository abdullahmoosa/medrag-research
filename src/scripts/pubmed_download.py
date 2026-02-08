from pathlib import Path
import json, time, ssl, urllib.request, urllib.error
from Bio import Entrez
from tqdm import tqdm

# ──────────────── CONFIG ─────────────────────────────────────
Entrez.email   = "nusratsultana@cuet.ac.bd"
Entrez.api_key = None              # add your NCBI key for higher quota
MAX_ARTICLES   = 100               # per subject

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
print(f"Project root: {PROJECT_ROOT}")

OUT_DIR        = PROJECT_ROOT / "data" / "pubmed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [
    "Anatomy","Biochemistry","Surgery","Ophthalmology","Physiology",
    "Social & Preventive Medicine","Gynaecology & Obstetrics","Anaesthesia",
    "Psychiatry","Microbiology","Medicine","Pharmacology","Dental","ENT",
    "Forensic Medicine","Pediatrics","Orthopaedics","Radiology","Pathology",
    "Skin","Unknown"
]

# ────────────────  GLOBAL SSL PATCH  ─────────────────────────
_unverified_ctx = ssl._create_unverified_context()
opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=_unverified_ctx)
)
urllib.request.install_opener(opener)
# ─────────────────────────────────────────────────────────────


def search_pmids(term, n, retries=3):
    query = f'("{term}"[MeSH Terms]) OR ("{term}"[Title/Abstract])'
    for attempt in range(retries):
        try:
            with Entrez.esearch(db="pubmed", term=query,
                                retmax=n, sort="relevance") as h:
                return Entrez.read(h)["IdList"]
        except urllib.error.URLError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    return []


def fetch_records(pmids, retries=3):
    ids = ",".join(pmids)
    for attempt in range(retries):
        try:
            with Entrez.efetch(db="pubmed", id=ids,
                               rettype="abstract", retmode="xml") as h:
                return Entrez.read(h)
        except urllib.error.URLError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    return {}


def xml_to_json(article):
    m   = article["MedlineCitation"]
    art = m["Article"]
    return {
        "pmid": m["PMID"],
        "title": art.get("ArticleTitle", ""),
        "abstract": " ".join(art.get("Abstract", {}).get("AbstractText", [])),
        "journal": art["Journal"]["Title"],
        "pub_year": art["Journal"]["JournalIssue"]["PubDate"].get("Year")
    }


def main():
    for subj in tqdm(SUBJECTS, desc="Subjects"):
        try:
            pmids = search_pmids(subj, MAX_ARTICLES)
            if not pmids:
                print(f"⚠️  No PMIDs found for {subj}")
                continue

            records  = fetch_records(pmids)
            outfile  = OUT_DIR / f"{subj.replace(' ', '_')}.jsonl"

            with outfile.open("w", encoding="utf-8") as fh:
                for art in records.get("PubmedArticle", []):
                    json.dump(xml_to_json(art), fh, ensure_ascii=False)
                    fh.write("\n")

            print(f"✅  Saved {len(records.get('PubmedArticle', []))} → "
                  f"{outfile.relative_to(PROJECT_ROOT)}")
            time.sleep(0.4)          # polite throttle
        except Exception as err:
            print(f"❌  {subj}: {err}")


if __name__ == "__main__":
    main()
