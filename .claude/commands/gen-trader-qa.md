# gen-trader-qa

Generate a two-trader Q&A document from any outline. Subagents write directly to temp files — no context overflow.

## Usage

```
/gen-trader-qa <outline_file> <output_file>
```

If no arguments given, ask the user for: path to outline file and output filename.

---

## Pipeline

### Step 0 — Check for outline file (you do this, do NOT delegate)

Check whether the outline file exists:

```bash
test -f <outline_file> && echo "exists" || echo "missing"
```

**If the outline file is missing**, generate it before continuing:

1. Ask the user: "What topics or themes should this Q&A cover?" (or infer from the output filename if obvious)
2. Spawn a single subagent to generate the outline and write it directly to `<outline_file>`:

```
Generate a Q&A outline document and write it to <outline_file> using the Write tool.

The outline should cover: [topics/themes from user or inferred from filename]

Format each theme as:

### Theme N — [Theme Name]

Q[N]. [Question text]
- T1: [one-line seed for Tactical Opportunist answer]
- T2: [one-line seed for Structured Growth Investor answer]

Aim for 6-8 questions per theme, 4-8 themes total (50 questions max).
Number questions sequentially from Q1. Do not return any text — write directly to the file.
```

3. Confirm the file was written, then continue to Step 1.

---

### Step 1 — Read context (you do this, do NOT delegate)

1. Read the outline file — extract all questions grouped by theme
2. Read the first 60 lines of the most recent completed doc in the same directory — for formatting reference only
3. Split questions into two halves by theme (e.g. themes 1-4 = part1, themes 5-8 = part2)

### Step 2 — Spawn 4 subagents IN PARALLEL

One subagent per (trader × part). All four run simultaneously.

| Subagent | Trader | Questions | Writes to |
|----------|--------|-----------|-----------|
| A | Trader 1 | Part 1 (first half of themes) | `/tmp/qa_t1_part1.md` |
| B | Trader 1 | Part 2 (second half of themes) | `/tmp/qa_t1_part2.md` |
| C | Trader 2 | Part 1 (first half of themes) | `/tmp/qa_t2_part1.md` |
| D | Trader 2 | Part 2 (second half of themes) | `/tmp/qa_t2_part2.md` |

**Every subagent prompt must include:**

```
You are writing [Trader 1/2] answers for a financial Q&A series.

[Paste the trader's profile summary — 5-8 bullet points on voice, holdings, style]

For each question below, write a 3-5 sentence answer in first-person matching that trader's voice.

IMPORTANT: Do NOT return your answers as text. Use the Write tool to write ALL answers
directly to [/tmp/qa_t1_part1.md]. Format:

**Q[N]. [question text]**

**[Trader name]:** [answer]

---

Questions:
[paste the questions for this part]
```

Key rules for the prompt:
- Include the full question text, not just the abbreviated outline seed
- The seed phrase is a starting hint — the subagent should expand it
- Tell the subagent explicitly: write to file, do not return text

### Step 3 — Assemble (you do this with a single Bash/Python call)

```python
import re, glob

# Read all 4 temp files
files = {
    't1': ['/tmp/qa_t1_part1.md', '/tmp/qa_t1_part2.md'],
    't2': ['/tmp/qa_t2_part1.md', '/tmp/qa_t2_part2.md'],
}

def parse_answers(paths):
    answers = {}
    for path in paths:
        try:
            text = open(path).read()
        except FileNotFoundError:
            continue
        # Split on Q number pattern
        blocks = re.split(r'\*\*Q(\d+)\.', text)
        for i in range(1, len(blocks), 2):
            qnum = int(blocks[i])
            answers[qnum] = blocks[i+1].strip()
    return answers

t1 = parse_answers(files['t1'])
t2 = parse_answers(files['t2'])

# Build interleaved doc — iterate in Q number order
all_qs = sorted(set(t1.keys()) | set(t2.keys()))

lines = [header]  # set header from outline metadata
for q in all_qs:
    lines.append(f'**Q{q}.** ...')  # question text from outline
    lines.append('')
    lines.append(f'**Trader 1:** {t1.get(q, "[missing]")}')
    lines.append('')
    lines.append(f'**Trader 2:** {t2.get(q, "[missing]")}')
    lines.append('')
    lines.append('---')
    lines.append('')

open(output_path, 'w').write('\n'.join(lines))
```

### Step 4 — Validate and clean up

- Print count of Q numbers found per trader and any gaps
- `rm /tmp/qa_t1_part1.md /tmp/qa_t1_part2.md /tmp/qa_t2_part1.md /tmp/qa_t2_part2.md`
- Report final file path and size

---

## Why this works

- Each subagent handles ~25 questions max — output stays under 30KB, well within limits
- Writing to files means zero text is returned through the agent result channel
- 4 subagents run in parallel — total time = time of slowest subagent, not sum of all
- Assembly is pure string parsing — fast, deterministic, no LLM needed
