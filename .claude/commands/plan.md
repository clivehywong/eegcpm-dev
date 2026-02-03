# Multi-Agent Planning Session

Initiate a multi-agent planning discussion for: $ARGUMENTS

## Instructions

You are initiating a **multi-agent planning session** using the centralized manager pattern.

### Agent Team
- **Architect** (Opus 4.5) - Lead, active in debate, final decisions
- **Researcher** (Gemini 3 Pro) - Deep analysis, options exploration
- **Critic** (GPT-5.1) - Validation, challenge assumptions

### Execution Flow

**Phase 1: Proposal**
1. Spawn Researcher agent (gemini/gemini-3-pro-preview) to analyze the task and propose approaches
2. Spawn Critic agent (openai/gpt-5.1) to assess the Researcher's proposal

**Phase 2: Debate**
3. As Architect, review both inputs and actively engage:
   - Challenge weak assumptions
   - Contribute your own ideas
   - Ask probing questions
4. Allow one round of responses from Researcher and Critic

**Phase 3: Synthesis**
5. Synthesize all inputs into a final decision
6. Output the plan with clear next steps

### Output Format

```
## Planning Session: {topic}

### Phase 1: Proposals

**Researcher (Gemini 3 Pro):**
{researcher output}

**Critic (GPT-5.1):**
{critic output}

### Phase 2: Debate

**Architect Engagement:**
{your challenges, questions, ideas}

**Responses:**
{agent responses}

### Phase 3: Decision

**Final Plan:**
{synthesized decision}

**Rationale:**
{why this approach}

**Next Steps:**
1. {step}
2. {step}
...
```

Begin the planning session now.
