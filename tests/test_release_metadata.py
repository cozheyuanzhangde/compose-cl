"""Keep the public README, citation, and local documentation links consistent."""

from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml


ROOT = Path(__file__).resolve().parents[1]
PAPER_TITLE = "Continual Learning Mechanisms Compose for Long-Horizon Memorization"


def test_readme_and_citation_agree():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    paper = citation["preferred-citation"]
    assert readme.startswith("# ComposeCL\n")
    assert "https://compose-cl.github.io/" in readme
    assert "git clone https://github.com/cozheyuanzhangde/compose-cl.git" in readme
    assert citation["repository-code"] == "https://github.com/cozheyuanzhangde/compose-cl"
    assert citation["url"] == "https://compose-cl.github.io/"
    assert citation["title"] == "ComposeCL"
    assert paper["title"] == PAPER_TITLE
    assert paper["type"] == "article"
    assert paper["year"] == 2026
    assert paper["authors"] == citation["authors"]
    assert [(author["given-names"], author["family-names"]) for author in paper["authors"]] == [
        ("Zheyuan", "Zhang"), ("Alvin", "Zhang"),
        ("Daniel", "Khashabi"), ("Tianmin", "Shu"),
    ]
    bibtex = re.search(r"```bibtex\n(.*?)\n```", readme, re.DOTALL).group(1)
    assert bibtex.startswith("@article{zhang2026continual,")
    assert f"title  = {{{PAPER_TITLE}}}" in bibtex
    assert not re.search(r"^\s*url\s*=", bibtex, re.MULTILINE)


def test_readme_local_links_resolve():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = {
        re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        for heading in re.findall(r"^#+ (.+)$", readme, re.MULTILINE)
    }
    for destination in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme):
        url = urlsplit(destination)
        if url.scheme or url.netloc:
            continue
        if url.path:
            assert (ROOT / url.path).is_file(), destination
        elif url.fragment:
            assert url.fragment in headings, destination
