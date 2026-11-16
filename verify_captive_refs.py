import json, time, urllib.parse, urllib.request

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0"})
    for a in range(6):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (a + 1)); continue
            return {}
        except Exception:
            time.sleep(3)
    return {}

targets = [
    ("Mian & Smith 1992", "Accounts receivable management policy theory and evidence", "Mian"),
    ("Caniato et al. 2016", "Does finance solve the supply chain financing problem", "Caniato"),
]

for label, title, author in targets:
    url = ('https://api.crossref.org/works?query.title=' + urllib.parse.quote(title)
           + '&query.author=' + author + '&rows=5&mailto=hxpnj@seu.edu.cn')
    d = get(url)
    print('===', label, '===')
    for it in d.get('message', {}).get('items', [])[:4]:
        t = (it.get('title') or ['?'])[0][:70]
        j = (it.get('container-title') or ['?'])[0]
        au = ", ".join(a.get('family', '?') for a in it.get('author', [])[:5])
        yr = None
        for k in ('published-print', 'published-online', 'issued'):
            dd = it.get(k)
            if dd and dd.get('date-parts'):
                yr = dd['date-parts'][0][0]; break
        print(f"  [{it.get('type')}] {yr} | {j[:32]:32s} | v{it.get('volume')} "
              f"p{it.get('page')} | {t} | {au} | {it.get('DOI')}")
    time.sleep(1.2)
