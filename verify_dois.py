"""
Precise verification via known DOIs and exact-title queries for the papers
that the bibliographic search mis-matched. Outputs corrected metadata.
"""
import json
import time
import urllib.parse
import urllib.request

MAILTO = "hxpnj@seu.edu.cn"
HDR = {"User-Agent": "ref-check/1.0"}


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
    return {"error": "persistent 429"}


# (label, DOI)  -- DOIs I am confident about
KNOWN_DOIS = [
    ("Barabasi & Albert 1999 (Science)", "10.1126/science.286.5439.509"),
    ("Watts & Strogatz 1998 (Nature)", "10.1038/30918"),
    ("Gao Barzel Barabasi 2016 (Nature)", "10.1038/nature16948"),
    ("Acemoglu Ozdaglar Tahbaz-Salehi 2015 (AER)", "10.1257/aer.20130456"),
    ("Elliott Golub Jackson 2014 (AER)", "10.1257/aer.104.10.3115"),
    ("Gai & Kapadia 2010 (Proc R Soc A)", "10.1098/rspa.2009.0410"),
    ("Yang & Birge 2018 (Management Science)", "10.1287/mnsc.2017.2795"),
    ("Tang Yang Wu 2018 (MSOM)", "10.1287/msom.2017.0676"),
    ("Klapper Laeven Rajan 2012 (RFS)", "10.1093/rfs/hhs006"),
    ("Barrot 2016 (JF)", "10.1111/jofi.12385"),
    ("Breza & Liberman 2018 (JF)", "10.1111/jofi.12589"),
    ("Battiston et al. 2012 (Sci Rep)", "10.1038/srep00541"),
    ("Deng Gu Cai Li 2018 (MSOM)", "10.1287/msom.2017.0677"),
]


def fmt(msg):
    authors = ", ".join(
        (a.get("family", "?") + " " + (a.get("given", "") or "")[:1]).strip()
        for a in msg.get("author", [])[:8])
    year = None
    for key in ("published-print", "published-online", "issued"):
        d = msg.get(key)
        if d and d.get("date-parts"):
            year = d["date-parts"][0][0]
            break
    return {
        "doi": msg.get("DOI"),
        "title": (msg.get("title") or ["?"])[0],
        "journal": (msg.get("container-title") or ["?"])[0],
        "year": year, "volume": msg.get("volume"),
        "issue": msg.get("issue"), "page": msg.get("page"),
        "authors": authors,
    }


out = {}
for label, doi in KNOWN_DOIS:
    d = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    msg = d.get("message")
    if msg:
        out[label] = {"status": "found", "meta": fmt(msg)}
    else:
        out[label] = {"status": "NOT_FOUND", "doi": doi, "err": d.get("error")}
    time.sleep(1.0)

with open("/home/xiaoping/Captive Factoring and Supply network Resilience/results/ref_verification_doi.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"{'label':40s} {'year':5s} {'vol':6s} {'page':12s} journal")
print("-" * 100)
for label, r in out.items():
    if r["status"] == "found":
        m = r["meta"]
        print(f"{label:40s} {str(m['year']):5s} {str(m['volume']):6s} "
              f"{str(m['page']):12s} {m['journal'][:30]}")
    else:
        print(f"{label:40s} {r['status']}")
