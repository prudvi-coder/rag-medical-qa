import os
import json
import time
from collections import Counter
from dotenv import load_dotenv
from Bio import Entrez

load_dotenv()

# --- Config ---
Entrez.email = "prudvisep@gmail.com"
Entrez.api_key = os.getenv("NCBI_API_KEY")

SEARCH_TERMS = [
    "Crohn's disease treatment",
    "Crohn's disease biologics",
    "inflammatory bowel disease diagnosis",
    "IBD immunosuppressants",
    "Crohn's disease clinical trial",
    "Crohn's disease symptoms pathophysiology",
    "mechanism of action biologics IBD",
    "anti-TNF therapy inflammatory bowel disease",
    "Crohn's disease review",
]

OUTPUT_FILE = "data/raw/pubmed_abstracts.json"
MAX_PER_QUERY = 80


def search_pubmed(query, max_results):
    """Search PubMed and return list of PMIDs."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstracts(pmid_list):
    """Fetch full abstract details for a list of PMIDs."""
    ids = ",".join(pmid_list)
    handle = Entrez.efetch(
        db="pubmed",
        id=ids,
        rettype="xml",
        retmode="xml"
    )
    records = Entrez.read(handle)
    handle.close()
    return records


def parse_records(records):
    """Parse XML records into clean dicts."""
    parsed = []
    for article in records["PubmedArticle"]:
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]

            # Title
            title = str(art.get("ArticleTitle", "")).strip()

            # Abstract
            abstract_obj = art.get("Abstract", {})
            abstract_texts = abstract_obj.get("AbstractText", [])
            if isinstance(abstract_texts, list):
                abstract = " ".join(str(t) for t in abstract_texts).strip()
            else:
                abstract = str(abstract_texts).strip()

            # Skip if no abstract
            if not abstract or len(abstract) < 50:
                continue

            # PMID
            pmid = str(medline["PMID"])

            # Journal
            journal = str(art["Journal"]["Title"])

            # Year
            pub_date = art["Journal"]["JournalIssue"]["PubDate"]
            year = str(pub_date.get("Year", pub_date.get("MedlineDate", "Unknown")))[:4]

            parsed.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        except Exception as e:
            print(f"  Skipping malformed record: {e}")
            continue

    return parsed


def main():
    all_abstracts = []
    seen_pmids = set()

    for query in SEARCH_TERMS:
        print(f"\nSearching: '{query}'")
        pmids = search_pubmed(query, MAX_PER_QUERY)
        print(f"  Found {len(pmids)} PMIDs")

        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)
        print(f"  New (non-duplicate): {len(new_pmids)}")

        if not new_pmids:
            continue

        for i in range(0, len(new_pmids), 50):
            batch = new_pmids[i:i+50]
            print(f"  Fetching batch {i//50 + 1}...")
            try:
                records = fetch_abstracts(batch)
                parsed = parse_records(records)
                all_abstracts.extend(parsed)
                print(f"  Parsed {len(parsed)} abstracts")
            except Exception as e:
                print(f"  Batch failed: {e}")
            time.sleep(0.5)

    # Save to JSON
    os.makedirs("data/raw", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_abstracts, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Total abstracts saved: {len(all_abstracts)}")
    print(f"Saved to: {OUTPUT_FILE}")

    # Data quality report
    lengths = [len(a["abstract"]) for a in all_abstracts]
    years = [a["year"] for a in all_abstracts]
    journals = [a["journal"] for a in all_abstracts]

    print(f"\n--- Data Quality Report ---")
    print(f"Total abstracts: {len(all_abstracts)}")
    print(f"Avg abstract length: {sum(lengths)//len(lengths)} chars")
    print(f"Year range: {min(years)} - {max(years)}")
    print(f"Top 5 journals:")
    for journal, count in Counter(journals).most_common(5):
        print(f"  {count}x {journal}")


if __name__ == "__main__":
    main()