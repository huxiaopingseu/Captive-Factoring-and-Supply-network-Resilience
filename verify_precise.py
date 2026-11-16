"""
Precise author+title queries for papers whose bibliographic search or my DOI
guess was wrong. Prints candidates for manual verification.
"""
import json
import time
import urllib.parse
import urllib.request

MAILTO = "hxpnj@seu.edu.cn"
HDR = {"User-Agent": "ref-check/1.0"}

# (label, exact title, first-author surname, expected journal fragment)
TARGETS = [
    ("Kouvelis & Xu 2021", "A Supply Chain Theory of Factoring and Reverse Factoring", "Kouvelis", "Management Science"),
    ("Tanrisever et al.", "Value of Reverse Factoring in Multi-Stage Supply Chains", "Tanrisever", "SSRN"),
    ("Yang & Birge 2018", "Trade Credit, Risk Sharing, and Inventory Financing Portfolios", "Yang", "Management Science"),
    ("Tang Yang Wu 2018", "Sourcing from Suppliers with Financial Constraints and Performance Risk", "Tang", "Manufacturing"),
    ("Klapper Laeven Rajan 2012", "Trade Credit Contracts", "Klapper", "Review of Financial Studies"),
    ("Cachon 2003", "Supply Chain Coordination with Contracts", "Cachon", "Handbooks"),
    ("Christopher & Peck 2004", "Building the Resilient Supply Chain", "Christopher", "Logistics"),
    ("Hosseini Ivanov Dolgui 2019", "Review of quantitative methods for supply chain resilience", "Hosseini", "Transportation Research"),
]


def get(url):
    req = urllib.request.Request(url, headers=HDR)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return {"error": str(e)}
        except Exception as e:
            time.sleep(3)
    return {"error": "429"}


for label, title, author, journal_frag in TARGETS:
    url = ("https://api.crossref.org/works?query.title="
           + urllib.parse.quote(title)
           + "&query.author=" + urllib.parse.quote(author)
           + "&rows=6&mailto=" + MAILTO)
    d = get(url)
    items = d.get("message", {}).get("items", [])
    print(f"\n=== {label} ===")
    for it in items:
        t = (it.get("title") or ["?"])[0][:60]
        j = (it.get("container-title") or ["?"])[0]
        au = ", ".join(a.get("family", "?") for a in it.get("author", [])[:4])
        yr = None
        for k in ("published-print", "published-online", "issued"):
            dd = it.get(k)
            if dd and dd.get("date-parts"):
                yr = dd["date-parts"][0][0]; break
        print(f"  [{it.get('type')}] {yr} | {j[:30]:30s} | v{it.get('volume')} "
              f"p{it.get('page')} | {t} | {au} | {it.get('DOI')}")
    time.sleep(1.2)
