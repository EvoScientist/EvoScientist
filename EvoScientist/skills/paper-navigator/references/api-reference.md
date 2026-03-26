# API Reference

Quick reference for APIs used by paper-navigator scripts.

## Semantic Scholar (S2) API

**Base URL:** `https://api.semanticscholar.org/graph/v1`

**Auth:** Optional `S2_API_KEY` env var (header: `x-api-key`)
- Without key: ~100 requests / 5 minutes
- With key: ~1 request / second sustained

### Paper ID Formats

| Format | Example | Usage |
|--------|---------|-------|
| S2 ID | `649def34f8be52c8b66281af98ae884c09aef38b` | Default internal ID |
| ArXiv | `ArXiv:1706.03762` | Prefix with `ArXiv:` |
| DOI | `DOI:10.18653/v1/N18-3011` | Prefix with `DOI:` |
| ACL | `ACL:N18-3011` | Prefix with `ACL:` |
| PubMed | `PMID:19872477` | Prefix with `PMID:` |
| URL | `https://arxiv.org/abs/1706.03762` | Full URL also accepted |

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/paper/search?query=...&fields=...&limit=N` | GET | Search papers |
| `/paper/{id}?fields=...` | GET | Single paper details |
| `/paper/{id}/citations?fields=...&limit=N` | GET | Papers citing this paper |
| `/paper/{id}/references?fields=...&limit=N` | GET | Papers this paper cites |
| `/author/search?query=...` | GET | Search authors |
| `/author/{id}/papers?fields=...` | GET | Author's papers |

### Recommendations API

**Base URL:** `https://api.semanticscholar.org/recommendations/v1`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/papers/` | POST | `{"positivePaperIds": [...], "negativePaperIds": [...]}` |

### Useful Fields

```
paperId,externalIds,title,authors,year,citationCount,
influentialCitationCount,tldr,isOpenAccess,openAccessPdf,
publicationVenue,abstract,publicationDate
```

---

## arXiv API

**Base URL:** `http://export.arxiv.org/api/query`

**Auth:** None required
**Rate limit:** Max 1 request per 3 seconds (courtesy)

### Query Parameters

| Param | Example | Description |
|-------|---------|-------------|
| `search_query` | `cat:cs.CL AND ti:transformer` | Search query |
| `sortBy` | `submittedDate` / `relevance` | Sort order |
| `sortOrder` | `descending` / `ascending` | Sort direction |
| `max_results` | `50` | Limit (max 2000) |

### Query Syntax

- `ti:` — title
- `abs:` — abstract
- `au:` — author
- `cat:` — category
- `submittedDate:` — date range `[YYYYMMDD0000 TO YYYYMMDD2359]`
- Boolean: `AND`, `OR`, `ANDNOT`
- Exact phrase: `"chain of thought"`

### Response Format

Returns Atom XML. Namespace: `{http://www.w3.org/2005/Atom}`

---

## Jina Reader

**URL pattern:** `https://r.jina.ai/{target_url}`

**Auth:** Optional `JINA_API_KEY` (header: `Authorization: Bearer <key>`)
**Header:** `Accept: text/markdown`

Returns the target URL content as clean Markdown. Works with:
- arXiv abstract pages (HTML → Markdown)
- PDF URLs (extracts text)
- Any web page

---

## Papers With Code API

**Base URL:** `https://paperswithcode.com/api/v1`

**Auth:** None required
**Rate limit:** Generous (undocumented)

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/papers/?q=...` | GET | Search papers |
| `/papers/?arxiv_id=...` | GET | Find paper by arXiv ID |
| `/papers/{id}/repositories/` | GET | Code repos for a paper |
| `/tasks/?q=...` | GET | Search tasks/benchmarks |
| `/tasks/{id}/datasets/` | GET | Datasets for a task |
| `/evaluations/?task=...&dataset=...` | GET | SOTA results |
| `/datasets/?q=...` | GET | Search datasets |

### IDs

Paper IDs and task IDs use URL-slug format (e.g. `attention-is-all-you-need`, `machine-translation`).
