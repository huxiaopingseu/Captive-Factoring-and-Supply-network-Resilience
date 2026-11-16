"""
Verify candidate references against Crossref (real, accurate metadata).
Uses the polite pool (mailto) with retry on 429 and per-query delay.
"""
import json
import time
import urllib.parse
import urllib.request

CANDIDATES = [
    ("Kouvelis & Xu 2021", "A Supply Chain Theory of Factoring and Reverse Factoring"),
    ("Tanrisever et al.", "Value of reverse factoring in multi-stage supply chains"),
    ("van der Vliet et al. 2015", "The price of reverse factoring financing rates versus payment delays"),
    ("Zhao & Huchzermeier 2018", "Managing supplier financial distress with advance payment discount and purchase order financing"),
    ("Yang & Birge 2018", "Trade credit risk sharing and inventory financing portfolios"),
    ("Tang Yang Wu 2018", "Sourcing from suppliers with financial constraints and performance risk"),
    ("Deng Gu Cai Li 2018", "Financing multiple heterogeneous suppliers in assembly systems buyer finance"),
    ("Babich Burnetas Ritchken 2007", "Competition and diversification effects in supply chains with supplier default risk"),
    ("Kouvelis & Zhao 2012", "Financing the newsvendor supplier versus bank and the structure of optimal trade credit contracts"),
    ("Klapper Laeven Rajan 2012", "Trade credit contracts"),
    ("Devalkar & Krishnan 2019", "The impact of working capital financing costs on the efficiency of trade credit"),
    ("Chod Trichakis et al. 2020", "On the financing benefits of supply chain transparency and blockchain adoption"),
    ("Cachon 2003", "Supply chain coordination with contracts"),
    ("Christopher & Peck 2004", "Building the resilient supply chain"),
    ("Pettit Fiksel Croxton 2010", "Ensuring supply chain resilience development of a conceptual framework"),
    ("Tukamuhabwa et al. 2015", "Supply chain resilience definition review and theoretical foundations for further study"),
    ("Hosseini Ivanov Dolgui 2019", "Review of quantitative methods for supply chain resilience"),
    ("Ivanov & Dolgui 2020", "Viability of intertwined supply networks extending the supply chain resilience angles towards survivability"),
    ("Barabasi & Albert 1999", "Emergence of scaling in random networks"),
    ("Watts & Strogatz 1998", "Collective dynamics of small-world networks"),
    ("Gao Barzel Barabasi 2016", "Universal resilience patterns in complex networks"),
    ("Acemoglu Ozdaglar Tahbaz-Salehi 2015", "Systemic risk and stability in financial networks"),
    ("Elliott Golub Jackson 2014", "Financial networks and contagion"),
    ("Gai & Kapadia 2010", "Contagion in financial networks"),
    ("Battiston et al. 2012", "DebtRank too central to fail financial networks the Federal Reserve and systemic risk"),
    ("Barrot 2016", "Trade credit and industry dynamics evidence from trucking firms"),
    ("Breza & Liberman 2018", "Financial contracting and organizational form evidence from the regulation of trade credit"),
]

MAILTO = "hxpnj@seu.edu.cn"


def crossref_query_title(title, rows=3):
    url = ("https://api.crossref.org/works?query.bibliographic="
           + urllib.parse.quote(title) + f"&rows={rows}&mailto={MAILTO}")
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(3)
    raise RuntimeError("persistent 429")


def fmt(item):
    # query.bibliographic returns flat work objects (not nested under "message")
    authors = ", ".join(
        (a.get("family", "?") + " " + (a.get("given", "") or "")[:1]).strip()
        for a in item.get("author", [])[:6])
    title = (item.get("title") or ["?"])[0][:70]
    journal = (item.get("container-title") or ["?"])[0]
    year = None
    for key in ("published-print", "published-online", "issued"):
        d = item.get(key)
        if d and d.get("date-parts"):
            year = d["date-parts"][0][0]
            break
    return {
        "doi": item.get("DOI"),
        "title": title,
        "journal": journal,
        "year": year,
        "volume": item.get("volume"),
        "issue": item.get("issue"),
        "page": item.get("page"),
        "type": item.get("type"),
        "authors": authors,
    }


def pick(items):
    """Prefer a journal-article match over working papers / reports."""
    for it in items:
        if it.get("type") == "journal-article" and (it.get("title") or [None])[0]:
            return it
    for it in items:
        if (it.get("title") or [None])[0]:
            return it
    return None


results = {}
for name, title in CANDIDATES:
    try:
        data = crossref_query_title(title)
        items = data.get("message", {}).get("items", [])
        best = pick(items)
        if best is not None:
            results[name] = {"status": "found", "top": fmt(best)}
        else:
            results[name] = {"status": "NOT_FOUND"}
    except Exception as e:
        results[name] = {"status": "ERROR", "error": str(e)}
    time.sleep(1.0)
    # incremental write so partial progress survives any interruption
    with open("/home/xiaoping/Captive Factoring and Supply network Resilience/results/ref_verification.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print(f"{'label':30s} {'year':5s} {'vol':6s} {'pg':12s}  title / journal")
print("-" * 110)
for name, r in results.items():
    if r["status"] == "found":
        m = r["top"]
        print(f"{name:30s} {str(m['year']):5s} {str(m['volume']):6s} "
              f"{str(m['page']):12s}  {m['title'][:38]} | {m['journal'][:22]}")
    else:
        print(f"{name:30s} {r['status']:12s} {r.get('error','')[:40]}")
