---
name: skill-creator
description: Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, update or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy.
---

# Skill Creator

A skill for creating new skills and iteratively improving them.

At a high level, the process of creating a skill goes like this:

- Decide what you want the skill to do and roughly how it should do it
- Write a draft of the skill
- Create a few test prompts and run the agent with access to the skill on them
- Help the user evaluate the results both qualitatively and quantitatively
  - While the runs happen in the background, draft some quantitative evals if there aren't any (if there are some, you can either use as is or modify if you feel something needs to change about them). Then explain them to the user (or if they already existed, explain the ones that already exist)
  - Use the `eval-viewer/generate_review.py` script to show the user the results for them to look at, and also let them look at the quantitative metrics
- Rewrite the skill based on feedback from the user's evaluation of the results (and also if there are any glaring flaws that become apparent from the quantitative benchmarks)
- Repeat until you're satisfied
- Expand the test set and try again at larger scale

Your job when using this skill is to figure out where the user is in this process and then jump in and help them progress through these stages.

## Communicating with the user

The skill creator is liable to be used by people across a wide range of familiarity with coding jargon. Pay attention to context cues to understand how to phrase your communication! In the default case:

- "evaluation" and "benchmark" are borderline, but OK
- for "JSON" and "assertion" you want to see serious cues from the user that they know what those things are before using them without explaining them

It's OK to briefly explain terms if you're in doubt.

---

## Skill Design Fundamentals

### About Skills

Skills are modular, self-contained packages that extend agent capabilities by providing
specialized knowledge, workflows, and tools. Think of them as "onboarding guides" for specific
domains or tasks -- they transform a general-purpose agent into a specialized agent
equipped with procedural knowledge and domain expertise.

#### What Skills Provide

1. Specialized workflows - Multi-step procedures for specific domains
2. Tool integrations - Instructions for working with specific file formats or APIs
3. Domain expertise - Company-specific knowledge, schemas, business logic
4. Bundled resources - Scripts, references, and assets for complex and repetitive tasks

### Concise is Key

The context window is a public good. Skills share the context window with everything else the agent needs: system prompt, conversation history, other Skills' metadata, and the actual user request.

**Default assumption: The agent is already very capable.** Only add context the agent doesn't already have. Challenge each piece of information: "Does the agent really need this explanation?" and "Does this paragraph justify its token cost?"

Prefer concise examples over verbose explanations.

### Set Appropriate Degrees of Freedom

Match the level of specificity to the task's fragility and variability:

**High freedom (text-based instructions)**: Use when multiple approaches are valid, decisions depend on context, or heuristics guide the approach.

**Medium freedom (pseudocode or scripts with parameters)**: Use when a preferred pattern exists, some variation is acceptable, or configuration affects behavior.

**Low freedom (specific scripts, few parameters)**: Use when operations are fragile and error-prone, consistency is critical, or a specific sequence must be followed.

Think of the agent as exploring a path: a narrow bridge with cliffs needs specific guardrails (low freedom), while an open field allows many routes (high freedom).

### Anatomy of a Skill

Every skill consists of a required SKILL.md file and optional bundled resources:

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter metadata (required)
│   │   ├── name: (required)
│   │   └── description: (required)
│   └── Markdown instructions (required)
└── Bundled Resources (optional)
    ├── scripts/          - Executable code (Python/Bash/etc.)
    ├── references/       - Documentation intended to be loaded into context as needed
    └── assets/           - Files used in output (templates, icons, fonts, etc.)
```

#### SKILL.md (required)

- **Frontmatter** (YAML): Contains `name` and `description` fields. These are the only fields that the agent reads to determine when the skill gets used, thus it is very important to be clear and comprehensive in describing what the skill is, and when it should be used.
- **Body** (Markdown): Instructions and guidance for using the skill. Only loaded AFTER the skill triggers (if at all).

#### Scripts (`scripts/`)

Executable code (Python/Bash/etc.) for tasks that require deterministic reliability or are repeatedly rewritten.

- **When to include**: When the same code is being rewritten repeatedly or deterministic reliability is needed
- **Benefits**: Token efficient, deterministic, may be executed without loading into context

#### References (`references/`)

Documentation and reference material intended to be loaded as needed into context.

- **When to include**: For documentation that the agent should reference while working
- **Best practice**: If files are large (>10k words), include grep search patterns in SKILL.md
- **Avoid duplication**: Information should live in either SKILL.md or references files, not both

#### Assets (`assets/`)

Files not intended to be loaded into context, but rather used within the output the agent produces.

- **When to include**: When the skill needs files that will be used in the final output (templates, images, fonts, etc.)

#### What to Not Include

Do NOT create extraneous documentation or auxiliary files (README.md, CHANGELOG.md, INSTALLATION_GUIDE.md, etc.). The skill should only contain information needed for an AI agent to do the job at hand.

### Progressive Disclosure

Skills use a three-level loading system to manage context efficiently:

1. **Metadata (name + description)** - Always in context (~100 words)
2. **SKILL.md body** - When skill triggers (<500 lines ideal)
3. **Bundled resources** - As needed (unlimited, scripts can execute without loading)

**Key patterns:**
- Keep SKILL.md under 500 lines; split into reference files when approaching this limit
- Reference files clearly from SKILL.md with guidance on when to read them
- For large reference files (>300 lines), include a table of contents

**Domain organization**: When a skill supports multiple domains/frameworks, organize by variant:
```
cloud-deploy/
├── SKILL.md (workflow + selection)
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```
The agent reads only the relevant reference file.

---

## Creating a Skill

### Capture Intent

Start by understanding the user's intent. The current conversation might already contain a workflow the user wants to capture (e.g., they say "turn this into a skill"). If so, extract answers from the conversation history first.

1. What should this skill enable the agent to do?
2. When should this skill trigger? (what user phrases/contexts)
3. What's the expected output format?
4. Should we set up test cases to verify the skill works?

### Interview and Research

Proactively ask questions about edge cases, input/output formats, example files, success criteria, and dependencies. Wait to write test prompts until you've got this part ironed out.

### Write the SKILL.md

Based on the user interview, fill in these components:

- **name**: Skill identifier (kebab-case)
- **description**: When to trigger, what it does. This is the primary triggering mechanism -- include both what the skill does AND specific contexts for when to use it. All "when to use" info goes here, not in the body. Make descriptions a little "pushy" to combat undertriggering.
- **compatibility**: Required tools, dependencies (optional)
- **the rest of the skill**

### Skill Writing Guide

**Writing style:** Use imperative form. Explain the **why** behind instructions rather than heavy-handed MUSTs. LLMs have good theory of mind -- transmit understanding, not just rules.

**Output format patterns**: See `references/output-patterns.md` for template and example patterns.

**Multi-step processes**: See `references/workflows.md` for sequential workflows and conditional logic.

### Implementing Resources

Start with the reusable resources identified during planning: `scripts/`, `references/`, and `assets/` files. Added scripts must be tested by actually running them.

### Test Cases

After writing the skill draft, come up with 2-3 realistic test prompts. Share them with the user for confirmation. Save test cases to `evals/evals.json`:

```json
{
  "skill_name": "example-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "User's task prompt",
      "expected_output": "Description of expected result",
      "files": []
    }
  ]
}
```

See `references/schemas.md` for the full schema (including the `assertions` field).

---

## Running and Evaluating Test Cases

This section is one continuous sequence. Put results in `<skill-name>-workspace/` as a sibling to the skill directory. Organize results by iteration (`iteration-1/`, `iteration-2/`, etc.) and within that, each test case gets a directory (`eval-0/`, `eval-1/`, etc.).

### Step 1: Spawn all runs (with-skill AND baseline) using `task` tool

For each test case, use the `task` tool to spawn two sub-agents -- one with the skill, one without. Launch everything at once so it all finishes around the same time.

**With-skill run:**

```
Execute this task:
- Skill path: <path-to-skill>
- Task: <eval prompt>
- Input files: <eval files if any, or "none">
- Save outputs to: <workspace>/iteration-<N>/eval-<ID>/with_skill/outputs/
- Outputs to save: <what the user cares about>
```

**Baseline run** (same prompt, but the baseline depends on context):
- **Creating a new skill**: no skill at all. Same prompt, no skill path, save to `without_skill/outputs/`.
- **Improving an existing skill**: the old version. Snapshot the skill first, then point the baseline at the snapshot. Save to `old_skill/outputs/`.

Write an `eval_metadata.json` for each test case (assertions can be empty for now):

```json
{
  "eval_id": 0,
  "eval_name": "descriptive-name-here",
  "prompt": "The user's task prompt",
  "assertions": []
}
```

### Step 2: While runs are in progress, draft assertions

Draft quantitative assertions for each test case and explain them to the user. Good assertions are objectively verifiable and have descriptive names. Subjective skills are better evaluated qualitatively.

Update the `eval_metadata.json` files and `evals/evals.json` with the assertions once drafted.

### Step 3: As runs complete, capture timing data

When each sub-agent task completes, save timing data to `timing.json`:

```json
{
  "total_tokens": 84852,
  "duration_ms": 23332,
  "total_duration_seconds": 23.3
}
```

### Step 4: Grade, aggregate, and launch the viewer

Once all runs are done:

1. **Grade each run** -- spawn a grader sub-agent that reads `agents/grader.md` and evaluates each assertion. Save results to `grading.json`. The grading.json expectations array must use fields `text`, `passed`, and `evidence`. For programmatic assertions, write and run a script.

2. **Aggregate into benchmark**:
   ```bash
   python -m scripts.aggregate_benchmark <workspace>/iteration-N --skill-name <name>
   ```
   This produces `benchmark.json` and `benchmark.md`.

3. **Do an analyst pass** -- read the benchmark data and surface patterns. See `agents/analyzer.md` for what to look for.

4. **Launch the viewer**:
   ```bash
   nohup python <skill-creator-path>/eval-viewer/generate_review.py \
     <workspace>/iteration-N \
     --skill-name "my-skill" \
     --benchmark <workspace>/iteration-N/benchmark.json \
     > /dev/null 2>&1 &
   VIEWER_PID=$!
   ```
   For iteration 2+, also pass `--previous-workspace <workspace>/iteration-<N-1>`.

   **Headless environments:** Use `--static <output_path>` to write a single-file HTML instead of starting a server. Note: the file still loads Google Fonts and SheetJS from CDN, so an internet connection is needed for full rendering.

5. **Tell the user** the results are ready in their browser.

### What the user sees in the viewer

The "Outputs" tab shows one test case at a time with prompt, output, previous output (iteration 2+), formal grades, and feedback textbox. The "Benchmark" tab shows stats summary. Navigation via prev/next or arrow keys. "Submit All Reviews" saves to `feedback.json`.

### Step 5: Read the feedback

When the user tells you they're done, read `feedback.json`. Empty feedback means the user thought it was fine. Focus improvements on test cases with specific complaints.

---

## Improving the Skill

### How to think about improvements

1. **Generalize from the feedback.** Don't overfit to specific examples. Rather than fiddly changes or oppressive MUSTs, try different metaphors or patterns.

2. **Keep the prompt lean.** Remove things that aren't pulling their weight. Read transcripts, not just outputs.

3. **Explain the why.** LLMs are smart. Transmit understanding into instructions rather than rigid rules.

4. **Look for repeated work across test cases.** If all test cases independently wrote similar helper scripts, bundle that script in `scripts/`.

### The iteration loop

After improving:
1. Apply improvements to the skill
2. Rerun all test cases into a new `iteration-<N+1>/` directory, including baselines
3. Launch the reviewer with `--previous-workspace`
4. Wait for user review
5. Read feedback, improve again, repeat

Keep going until the user is happy, feedback is empty, or you're not making progress.

---

## Advanced: Blind Comparison

For rigorous comparison between two versions, read `agents/comparator.md` and `agents/analyzer.md`. The idea: give two outputs to an independent agent without telling it which is which, and let it judge quality.

This is optional and most users won't need it.

---

## Description Optimization

The description field in SKILL.md frontmatter determines whether the agent invokes a skill. After creating or improving a skill, offer to optimize the description.

### Step 1: Generate trigger eval queries

Create 20 eval queries -- a mix of should-trigger and should-not-trigger. Save as JSON:

```json
[
  {"query": "the user prompt", "should_trigger": true},
  {"query": "another prompt", "should_trigger": false}
]
```

Queries must be realistic and specific. Focus on edge cases rather than clear-cut ones.

For **should-trigger** queries (8-10): different phrasings of the same intent, cases where the user doesn't explicitly name the skill but clearly needs it.

For **should-not-trigger** queries (8-10): near-misses that share keywords but need something different. Don't make them obviously irrelevant.

### Step 2: Review with user

Present the eval set using the HTML template:

1. Read `assets/eval_review.html`
2. Replace `__EVAL_DATA_PLACEHOLDER__`, `__SKILL_NAME_PLACEHOLDER__`, `__SKILL_DESCRIPTION_PLACEHOLDER__`
3. Write to a temp file and open it
4. User edits queries and exports the eval set

### Step 3: Run the optimization loop

Tell the user this will take some time. Save the eval set to the workspace, then run:

```bash
python -m scripts.run_loop \
  --eval-set <path-to-trigger-eval.json> \
  --skill-path <path-to-skill> \
  --model <model> --provider <provider> \
  --max-iterations 5 \
  --verbose
```

Use the model and provider from the user's EvoSci configuration so the triggering test matches their actual experience. Supports all 7+ providers (Anthropic, OpenAI, Google, NVIDIA, SiliconFlow, OpenRouter, Ollama, and custom endpoints).

This handles the full optimization loop automatically: splits eval set 60/40 train/test, evaluates current description (3 runs per query), proposes improvements, re-evaluates, iterates up to 5 times. Returns JSON with `best_description` selected by test score.

### How skill triggering works

Skills appear in the agent's system prompt with name + description. The agent decides whether to consult a skill based on that description. The agent only consults skills for tasks it can't easily handle on its own -- complex, multi-step, or specialized queries reliably trigger skills when the description matches.

### Step 4: Apply the result

Take `best_description` from the JSON output and update the skill's SKILL.md frontmatter. Show the user before/after and report scores.

---

## Installing and Packaging

### Install

Once the skill is ready, install it:

```
skill_manager(action="install", source="/<skill-name>")
```

### Verify

```
skill_manager(action="list")
```

### Validation checklist

- SKILL.md has valid YAML frontmatter with `name` and `description`
- Skill name is kebab-case (lowercase, digits, hyphens)
- Description clearly explains what the skill does and when to use it
- All referenced resource files exist
- No extraneous files (README.md, CHANGELOG.md, etc.)

You can also validate programmatically:

```bash
python -m scripts.quick_validate <path/to/skill>
```

### Initialize a new skill

```bash
python -m scripts.init_skill <skill-name> --path <path>
```

### Package for distribution

```bash
python -m scripts.package_skill <path/to/skill-folder>
```

---

## Reference Files

The `agents/` directory contains instructions for specialized sub-agents:
- `agents/grader.md` -- How to evaluate assertions against outputs
- `agents/comparator.md` -- How to do blind A/B comparison
- `agents/analyzer.md` -- How to analyze benchmark results

The `references/` directory has additional documentation:
- `references/schemas.md` -- JSON structures for evals.json, grading.json, benchmark.json
- `references/output-patterns.md` -- Output format template patterns
- `references/workflows.md` -- Sequential workflow and conditional logic patterns
