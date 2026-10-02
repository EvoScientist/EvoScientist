---
name: find-skills
description: Helps users discover agent skills from the open ecosystem. Searches skills.sh and presents options for installation via the built-in skill_manager tool.
---

# Find Skills

This skill helps you discover skills from the open agent skills ecosystem.

## When to Use This Skill

Use this skill when the user:

- Asks "how do I do X" where X might be a common task with an existing skill
- Says "find a skill for X" or "is there a skill for X"
- Wants to search for tools, templates, or workflows
- Expresses interest in extending agent capabilities
- Mentions they wish they had help with a specific domain (design, testing, deployment, etc.)

## Step 1: Search for Skills

Use `npx -y skills find` with a relevant keyword to search the ecosystem:

```bash
npx -y skills find [query]
```

Examples:
- User asks "help me with React performance" → `npx -y skills find react performance`
- User asks "is there a skill for PR reviews?" → `npx -y skills find pr review`
- User asks "I need to create a changelog" → `npx -y skills find changelog`

The search results will show installable skills like:

```
vercel-labs/agent-skills@vercel-react-best-practices
└ https://skills.sh/vercel-labs/agent-skills/vercel-react-best-practices
```

Browse all available skills at: https://skills.sh/

## Step 2: Present Options

When you find relevant skills, present them to the user with:
1. The skill name and what it does
2. A link to learn more on skills.sh

Ask the user which skill(s) they want to install.

## Step 3: Install

Use the built-in `skill_manager` tool to install:

```
skill_manager(action="install", source="owner/repo@skill-name")
```

## Common Skill Categories

| Category | Example Queries |
|----------|----------------|
| Web Development | react, nextjs, typescript, css, tailwind |
| Testing | testing, jest, playwright, e2e |
| DevOps | deploy, docker, kubernetes, ci-cd |
| Documentation | docs, readme, changelog, api-docs |
| Code Quality | review, lint, refactor, best-practices |
| Design | ui, ux, design-system, accessibility |
| Productivity | workflow, automation, git |

## When No Skills Are Found

If no relevant skills exist:

1. Acknowledge that no existing skill was found
2. Offer to help with the task directly using your general capabilities
3. Mention the user could create their own skill with `npx -y skills init`
