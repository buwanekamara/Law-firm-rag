# Demo questions

Questions for testing the system with the document filter set to **All
documents**. None of them name a contract, so nothing is filtered in advance and
retrieval has to pick the right clause out of all 120 chunks on its own.

Every expected section below was verified against the extracted text, not
guessed.

**How to read a result.** Look at the *retrieved* table first: if rank 1 is the
expected section, retrieval did its job. Then read the answer. If retrieval was
right and the answer is wrong, that is a generation problem, and
`?debug=true` shows exactly what the model was given.

---

## Questions with one right home

Each of these is answered by exactly one contract.

### Hosting Agreement

| Question | Expected |
|---|---|
| How much must the client pay in total, and how is it split between prepayment and completion? | Section 2 |
| Who is responsible for proofreading the text before it is published? | Section 1 |
| Is the company allowed to put its own credit on the finished site? | Section 1 |
| What are the normal working hours, and what happens if work is needed outside them? | Section 3 |

### Joint Venture Agreement

| Question | Expected |
|---|---|
| What is the name of the venture the two companies formed? | Section 1 |
| How are profits split between the two parties? | Section 6 |
| Where is the principal place of business located? | Section 1 / Section 4 |
| Can either party sell its interest to an outsider? | Section 12 |

### Manufacturing Agreement

| Question | Expected |
|---|---|
| Who carries the risk if goods are damaged in transit? | Section 8 |
| What happens if a batch has to be recalled? | Section 21 |
| Who pays legal costs if there is a dispute? | Section 19 |

### Trademark License Agreement

| Question | Expected |
|---|---|
| Can the licensee let a subsidiary use the brand? | Section 1.2 |
| Do both sides give up the right to a jury trial? | Section 9.5 |
| What must the licensee do before using the mark in any new material? | Section 3.2 |

### Transportation Agreement

| Question | Expected |
|---|---|
| What is a British thermal unit as defined here? | Article I |
| If capacity runs short, whose deliveries get cut first? | Article V |
| What security can be demanded from a customer who has not paid? | Article XIV |
| How must formal notices be delivered between the parties? | Article IX |

---

## Questions with more than one right home

These are answered by several contracts at once. A good answer covers each and
attributes every statement to its own agreement. **The failure to look for is
blending** — stating one contract's rule as though it applied to all of them, or
attaching a clause from one agreement to another's name.

| Question | Should draw on |
|---|---|
| Which state's law applies? | Manufacturing Section 12 (California), Trademark Section 9.4, Joint Venture Section 14 |
| What notice is required to terminate? | Trademark 4.2 / 4.3, Manufacturing Section 11 |
| What happens if a strike or a storm stops performance? | Manufacturing Section 10, Transportation Article X |
| What confidentiality obligations apply? | Manufacturing Section 9, and others in passing |

Verified citations make a fabricated *section* impossible — anything naming a
section that was not retrieved is stripped before you see it. What verification
cannot catch is the right section attached to the wrong claim, which is why
these questions are worth reading carefully.

---

## The questions designed to be got wrong

Five traps, each a different way for the honest answer to be "the contracts do
not give you this". These are the ones worth demonstrating, because getting them
wrong produces a confident, plausible, wrong answer rather than an obviously bad
one.

| Question | Correct behaviour |
|---|---|
| What is the price per unit under the manufacturing agreement? | Say the figure is **redacted** (`[***]`), cite Section 3. Not "not specified" — the clause is there, the number was removed before filing. |
| What is the effective date of the trademark licence? | Say the date was **left blank** (`[·]`) on the executed copy. Not to repeat the placeholder as though it were a date. |
| What gas quality specifications must the delivered gas meet? | Cite Article VII and say the specifications themselves are **not set out** — the clause defers to a standard we were not given. |
| Who owns the written material the client supplies for the site? | Say ownership is **not addressed**. Do not infer it from the licence granted in Section 1(d). |
| What is the notice period for terminating the office lease? | Refuse. There is no lease among these five contracts. |

And one more, which never reaches the model at all:

| Question | Correct behaviour |
|---|---|
| What is the capital of France? | Refused by the relevance gate — similarity about 0.47, below the 0.56 threshold. No model call is made. |

---

## Suggested order for a live demo

1. **How are profits split between the two parties?** — a clean hit, right section first.
2. **What notice is required to terminate?** — shows multi-document coverage with separate citations.
3. **What is the price per unit under the manufacturing agreement?** — the redaction trap.
4. **What is the notice period for terminating the office lease?** — a refusal where a plausible answer was available.
5. **What is the capital of France?** — the gate firing, with no model call.

That sequence takes about two minutes and covers retrieval, citation, both
kinds of refusal, and a guard doing something the prompt alone could not.
