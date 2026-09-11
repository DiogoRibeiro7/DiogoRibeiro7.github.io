"""Fetch freely licensed photographs from Wikimedia Commons as header images.

Files are taken when Commons reports their licence as CC0, public domain, or
a CC BY / CC BY-SA licence. Every pick is recorded with its author, licence
and source page in assets/images/headers/CREDITS.md and in
_data/image_credits.yml, which the /image-credits/ page renders, so the
attribution the CC BY licences require is published with the site.
Run from this directory:

    python fetch_headers.py            # every query below
    python fetch_headers.py circuit    # only queries whose slug contains "circuit"

Each query yields at most one image: the first search hit that is a JPEG at
least 1600 pixels wide, landscape, and free of the words that mark scans and
documents. The image is downloaded at 1600 pixels wide, centre-cropped to
16:9 and written as photo-<slug>.jpg.
"""
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

OUTDIR = Path(__file__).resolve().parent.parent / "images" / "headers"
CREDITS = OUTDIR / "CREDITS.md"
UA = "DiogoRibeiro7-site/1.0 (https://diogoribeiro7.github.io; hansolo.dj@gmail.com)"
API = "https://commons.wikimedia.org/w/api.php"
FREE = {"CC0", "Public domain", "CC0 1.0", "PD", "Public Domain"}
ATTRIBUTION = {"CC BY 4.0", "CC BY 3.0", "CC BY 2.5", "CC BY 2.0",
               "CC BY-SA 4.0", "CC BY-SA 3.0", "CC BY-SA 2.5", "CC BY-SA 2.0"}
DATA = Path(__file__).resolve().parent.parent.parent / "_data" / "image_credits.yml"
BAD_TITLE = re.compile(r"page|scan|document|manuscript|book|map of|logo|screenshot|abstract from|portrait", re.I)

QUERIES = {
    "lights":      "city light trails long exposure night",
    "geometry":    "geometric facade pattern architecture",
    "waves":       "ocean waves aerial view",
    "terrain":     "aerial river delta landscape",
    "library":     "reading room library interior",
    "earth":       "earth at night city lights satellite",
    "stars":       "milky way long exposure",
    "bridge":      "steel bridge structure",
    "code":        "source code on computer screen",
    "formulas":    "mathematical equations written",
    "fractal":     "mandelbrot fractal render",
    "dice":        "dice probability",
    "supercomputer": "supercomputer cabinets",
    "motherboard": "motherboard close-up",
    "wafer":       "silicon wafer",
    "punchcards":  "punched cards computer",
    "calculator":  "scientific calculator keys",
}


def get(params):
    params = dict(params, format="json")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def candidates(query, limit=25):
    d = get({"action": "query", "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
             "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
             "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1600})
    pages = d.get("query", {}).get("pages", {})
    for p in sorted(pages.values(), key=lambda x: x.get("index", 0)):
        ii = p["imageinfo"][0]
        em = ii.get("extmetadata", {})
        lic = (em.get("LicenseShortName", {}).get("value") or "").strip()
        yield {
            "title": p["title"], "width": ii["width"], "height": ii["height"], "mime": ii.get("mime", ""),
            "licence": lic, "author": re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value") or "").strip(),
            "thumb": ii.get("thumburl"), "page": ii.get("descriptionurl"),
        }


def acceptable(c):
    ratio = c["width"] / max(c["height"], 1)
    return (c["mime"] in ("image/jpeg", "image/png") and c["width"] >= 1600 and 1.3 <= ratio <= 2.4
            and (c["licence"] in FREE or c["licence"] in ATTRIBUTION)
            and not BAD_TITLE.search(c["title"]) and c["thumb"])


def crop_16_9(img):
    w, h = img.size
    target = w / (16 / 9)
    if h > target:
        top = int((h - target) / 2)
        img = img.crop((0, top, w, int(top + target)))
    else:
        new_w = int(h * 16 / 9)
        left = int((w - new_w) / 2)
        img = img.crop((left, 0, left + new_w, h))
    return img.resize((1600, 900), Image.LANCZOS)


def fetch(slug, query):
    for c in candidates(query):
        if not acceptable(c):
            continue
        req = urllib.request.Request(c["thumb"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        out = OUTDIR / f"photo-{slug}.jpg"
        crop_16_9(img).save(out, "JPEG", quality=85, optimize=True, progressive=True)
        c["file"] = out.name
        return c
    return None


def write_credits(picks):
    lines = ["# Header image credits", "",
             "Photographs fetched from Wikimedia Commons by `assets/viz/fetch_headers.py` under",
             "CC0, public-domain, CC BY or CC BY-SA licences; the same list is published at",
             "/image-credits/. The other files in this directory are generated by",
             "`assets/viz/generate_headers.py`.", "",
             "| File | Title | Author | Licence | Source |", "| --- | --- | --- | --- | --- |"]
    for c in picks:
        title = c["title"].replace("File:", "").replace("|", "/")
        lines.append(f"| {c['file']} | {title} | {c['author'] or 'unknown'} | {c['licence']} | [Commons]({c['page']}) |")
    CREDITS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    yml = ["# Written by assets/viz/fetch_headers.py; rendered by _pages/image-credits.md."]
    for c in picks:
        title = c["title"].replace("File:", "").replace('"', "'")
        author = (c["author"] or "unknown").replace('"', "'")
        yml += [f"- file: {c['file']}", f'  title: "{title}"', f'  author: "{author}"',
                f'  licence: "{c["licence"]}"', f'  source: "{c["page"]}"']
    DATA.write_text("\n".join(yml) + "\n", encoding="utf-8")


def main(filters):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if CREDITS.exists():
        for line in CREDITS.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\| (photo-[\w-]+\.jpg) \| (.*?) \| (.*?) \| (.*?) \| \[Commons\]\((.*?)\) \|", line)
            if m:
                existing.append({"file": m[1], "title": "File:" + m[2], "author": m[3], "licence": m[4], "page": m[5]})
    picks = {c["file"]: c for c in existing}
    for slug, query in QUERIES.items():
        if filters and not any(f in slug for f in filters):
            continue
        c = fetch(slug, query)
        if c:
            picks[c["file"]] = c
            print(f"  {c['file']:24} {c['width']}x{c['height']}  {c['licence']:14} {c['title'][:50]}")
        else:
            print(f"  {slug:24} no acceptable file for '{query}'")
    write_credits(sorted(picks.values(), key=lambda c: c["file"]))
    print(f"\n{len(picks)} photographs -> {OUTDIR}; credits in {CREDITS.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
