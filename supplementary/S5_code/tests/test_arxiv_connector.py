from __future__ import annotations

import httpx
import respx

from review_pipeline.src.search.connectors.arxiv import BASE_URL, search_arxiv_by_title

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2402.10977v1</id>
    <title>Generative AI and Process Systems Engineering: The Next Frontier</title>
    <summary>We discuss generative AI applications in process systems engineering.</summary>
    <published>2024-02-16T00:00:00Z</published>
    <author><name>Benjamin Decardi-Nelson</name></author>
    <author><name>Abdulelah S. Alshehri</name></author>
    <arxiv:journal_ref>Comput. Chem. Eng. (2024)</arxiv:journal_ref>
    <arxiv:doi>10.1016/j.compchemeng.2024.108723</arxiv:doi>
    <link title="pdf" href="http://arxiv.org/pdf/2402.10977v1" type="application/pdf"/>
  </entry>
</feed>
"""


@respx.mock
def test_search_arxiv_by_title_parses_entries() -> None:
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, text=SAMPLE_FEED))
    client = httpx.Client()
    results = search_arxiv_by_title(client, "Generative AI and Process Systems Engineering")

    assert len(results) == 1
    c = results[0]
    assert c.arxiv_id == "2402.10977v1"
    assert c.title == "Generative AI and Process Systems Engineering: The Next Frontier"
    assert c.author_surnames == ["Decardi-Nelson", "Alshehri"]
    assert c.year == 2024
    assert c.doi == "10.1016/j.compchemeng.2024.108723"
    assert c.pdf_url == "http://arxiv.org/pdf/2402.10977v1"


@respx.mock
def test_search_arxiv_by_title_zero_results() -> None:
    empty_feed = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, text=empty_feed))
    client = httpx.Client()
    assert search_arxiv_by_title(client, "nothing matches this") == []


@respx.mock
def test_search_arxiv_by_title_http_error_returns_empty_not_raise(monkeypatch) -> None:
    monkeypatch.setattr("review_pipeline.src.search.connectors.http_client.time.sleep", lambda _: None)
    respx.get(BASE_URL).mock(return_value=httpx.Response(500, text="error"))
    client = httpx.Client()
    assert search_arxiv_by_title(client, "anything") == []


@respx.mock
def test_search_arxiv_by_title_malformed_xml_returns_empty() -> None:
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, text="not xml at all <<<"))
    client = httpx.Client()
    assert search_arxiv_by_title(client, "anything") == []
