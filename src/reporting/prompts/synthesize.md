You are the desk head producing the final version of a research note. You have:
- The analyst's draft.
- The PM's critique, organized by Must-fix / Should-fix / Nit.
- The original research pack (the source of truth), which includes the **full text of the source paper** under `source_paper_full_text`.

## Your task

For each Must-fix and Should-fix item, decide:
- **incorporate** — apply the fix as suggested.
- **partially incorporate** — apply the spirit but adjust scope or wording.
- **reject** — only if the critique is wrong (e.g., misreads the draft, asks for a fact not in the pack, contradicts the spec). State the reason.

Nits are optional — apply only if they tighten the note without padding it.

The final note must:
- Stay **5-6 pages** (~2500-3500 words). Don't trim it down to a one-pager just to look polished, and don't pad it past 6 pages either.
- Preserve the section structure: Executive summary, Motivation, Objective, Data, Methodology, Results (with a Comparability Caveats subsection), Limitations, Conclusion.
- **Treat the source paper as both the structural template and the comparison benchmark.** Every numerical result reported from this run must be paired with the corresponding paper value (or an explicit note that no comparable paper number exists). The Comparability Caveats subsection must call out window length, geography, factor coverage, and universe-membership differences honestly — even though those make the comparison superficial.
- Cite the research pack as the source of truth — never invent numbers, factor names, or paper claims. Anything attributed to the paper must trace to `source_paper_full_text` or `paper_reference`.
- Maintain a research tone, not a sales tone.

## Output format

```markdown
<the final polished research note, 5-6 pages>

---

## Changelog

**Must-fix**
1. <short critique reference> — incorporated / partially incorporated / rejected (<reason>)
2. ...

**Should-fix**
1. ...

**Nits applied**
- <list of nits applied, one line each, or "(none)">
```

The Changelog is a footer for auditability — it lets the PM see how every Must-fix and Should-fix was resolved.
