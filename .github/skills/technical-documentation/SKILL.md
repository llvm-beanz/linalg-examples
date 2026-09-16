---
name: technical-documentation
description: Review, improve, and produce technical documents in clear, concise English. Use for specifications, design documents, tutorials, reference documentation, proposals, runbooks, READMEs, release notes, and other engineering documentation.
argument-hint: Describe the document, audience, purpose, and desired outcome.
---

# Technical Documentation Skill

Produce technical documents that are accurate, useful, scannable, and concise. Review existing documents for technical correctness and communication quality, or create new documents from requirements and authoritative source material.

## Core principles

- Optimize for the reader's task: make the purpose, outcome, and next action obvious.
- Prefer plain, precise English over jargon, idioms, filler, and unnecessarily formal language.
- State facts, assumptions, constraints, and decisions explicitly.
- Preserve technical meaning; never simplify by introducing an inaccurate claim.
- Use the fewest words that communicate the complete idea.
- Prefer active voice, concrete verbs, short sentences, and specific nouns.
- Make documents scannable with informative headings, lists, tables, examples, and code blocks.
- Treat commands, paths, identifiers, API names, and code as exact; do not paraphrase them.
- Do not invent behavior, metrics, citations, links, requirements, or implementation details.

## Workflow

### 1. Establish the document contract

Before writing or making substantial edits, determine:

- **Purpose:** What decision, task, or understanding should this document support?
- **Audience:** Who will read it, and what do they already know?
- **Scope:** What is included, excluded, and assumed?
- **Document type:** For example, specification, tutorial, reference, design proposal, runbook, README, or release note.
- **Success criteria:** What should a reader be able to do or decide afterward?
- **Source of truth:** Which code, configuration, tests, standards, or authoritative references support the content?

If important information is missing, ask focused questions. Otherwise, make conservative assumptions and label them.

### 2. Inspect and verify source material

When working in a repository:

1. Find related documentation and follow its local conventions.
2. Inspect the implementation, configuration, tests, examples, and standards that the document describes.
3. Verify names, paths, commands, options, prerequisites, defaults, versions, and expected outputs.
4. Identify contradictions between the request, existing documentation, and implementation.
5. Separate verified facts from interpretation and recommendations.

Do not claim that a command, example, or procedure works unless it is supported by the available source or has been run successfully.

### 3. Plan the information architecture

Choose the smallest structure that supports the reader's goal:

- Put the conclusion, purpose, or result first.
- Order instructions chronologically and explanations from general to specific.
- Put prerequisites before steps that depend on them.
- Introduce concepts before using their terminology.
- Keep one main idea per paragraph and one action per numbered step.
- Move detailed background, edge cases, and reproducibility data to later sections or appendices.

Use headings that describe content or outcomes, not vague labels such as "Details" or "Miscellaneous".

### 4. Draft or revise

For new documents, write a complete first draft with verified examples. For existing documents:

1. Preserve correct technical content and the author's intent.
2. Fix correctness and ambiguity before style.
3. Remove repetition, throat-clearing, redundant qualifiers, and unused sections.
4. Replace vague statements with measurable or observable language.
5. Split overloaded paragraphs and long sentences.
6. Make prerequisites, inputs, outputs, failure modes, and next steps explicit.
7. Keep examples minimal but realistic and ensure they match the surrounding explanation.

Do not rewrite merely to change style when the existing wording is already clear and accurate.

### 5. Review the result

Perform these passes in order:

1. **Technical accuracy:** Are claims, examples, procedures, and terminology supported by the source?
2. **Completeness:** Can the intended audience accomplish the stated goal without guessing?
3. **Organization:** Does the structure match the reader's workflow and put important information first?
4. **Clarity:** Are references, pronouns, conditions, dependencies, and ownership unambiguous?
5. **Conciseness:** Can any sentence, section, example, or qualifier be removed without losing meaning?
6. **Language quality:** Is the English grammatical, natural, consistent, and professional?
7. **Accessibility and scanability:** Are headings descriptive, lists parallel, tables justified, links meaningful, and code blocks labeled?

### 6. Report the outcome

For a review, provide:

- A brief overall assessment.
- Findings ordered by impact, with location, issue, rationale, and a specific suggested change.
- Separate correctness or usability issues from optional style improvements.
- Important assumptions and unverifiable items.
- A revised version only when requested or clearly useful.

For document production, provide the document itself and a short note identifying:

- The intended audience and purpose.
- The sources or assumptions used.
- Any unresolved questions, limitations, or follow-up work.

## English-language writing standards

### Prefer

- "Use X to do Y" instead of "In order to do Y, use X."
- "The command creates..." instead of "It can be seen that the command creates..."
- "If the build fails, check..." instead of "In the event that the build should fail..."
- Concrete subjects and verbs: "The service retries the request."
- Direct definitions on first use: "A manifest is a file that..."
- Consistent terms: choose one term for one concept and use it throughout.

### Avoid

- Empty openings: "This document will discuss..."
- Unqualified words such as "easy", "simple", "obviously", "just", and "always".
- Passive voice when it hides responsibility or the actor.
- Nominalizations when a direct verb is clearer: "make a decision" -> "decide".
- Multiple synonyms for the same concept.
- Marketing language, speculation presented as fact, and unexplained acronyms.
- Excessive parentheticals, nested clauses, exclamation marks, and decorative prose.

Use US English unless the project or user specifies another English variant. Preserve established product names, API spellings, and project terminology even when they differ from ordinary English.

## Document-type guidance

### Specifications and design documents

State the problem, goals, non-goals, terminology, requirements, constraints, interfaces, invariants, examples, compatibility impact, alternatives, and open questions. Distinguish normative requirements from informative explanation. Use precise terms such as **must**, **must not**, **should**, and **may** consistently, and define their meaning when the document does not inherit a standard.

### Tutorials and how-to guides

Lead with the outcome and prerequisites. Use a verified, minimal end-to-end example. Number actions in execution order, show expected results, explain why important steps matter, and include troubleshooting for likely failures.

### Reference documentation

Optimize for lookup. Describe behavior, syntax, parameters, defaults, constraints, errors, and compatibility precisely. Use tables only when they improve comparison or scanning.

### Runbooks

Make the document executable under pressure. Include symptoms, scope, prerequisites, safety checks, ordered commands, expected output, rollback or recovery steps, escalation criteria, and post-incident follow-up.

### READMEs and release notes

Put the fastest path first. Keep setup commands copyable. For releases, describe user-visible impact, breaking changes, migration actions, fixes, and known limitations; do not bury critical warnings.

## Formatting and examples

- Use Markdown that renders correctly in the target repository.
- Use fenced code blocks with a language identifier when applicable.
- Put placeholders in a visibly distinct form such as `<value>` and explain required substitutions.
- Use inline code for commands, paths, options, symbols, and literal values.
- Prefer tables for genuinely tabular information, not for prose.
- Use descriptive link text and verify links when tools are available.
- Keep examples self-contained enough to be useful, but omit irrelevant boilerplate.
- Mark illustrative or pseudocode examples clearly; never present them as verified commands.

## Final checklist

- [ ] Purpose, audience, scope, and outcome are clear.
- [ ] Important claims and examples are supported by authoritative sources.
- [ ] Prerequisites, inputs, outputs, constraints, and failure handling are covered where relevant.
- [ ] Normative requirements are distinguishable from informative explanation.
- [ ] The most useful information appears first.
- [ ] Headings and lists make the document easy to scan.
- [ ] Terminology, spelling, capitalization, formatting, and tense are consistent.
- [ ] Sentences are direct, concise, and unambiguous.
- [ ] No unsupported claims, dead links, unexplained acronyms, or misleading examples remain.
- [ ] Open questions and assumptions are explicitly identified.
