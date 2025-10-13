PROMPT_TEMPLATE_QWEN = """
Answer the given question. \
Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [Qwen2.5-7B-Instruct]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
You must not limit yourself to repeatedly calling a single LLM (unless its provided information is consistently the most effective and informative). \
You are encouraged to explore and utilize different LLMs to better understand their respective strengths and weaknesses. \
It is also acceptable—and recommended—to call different LLMs multiple times for the same input question to gather more comprehensive information. \


#### The Descriptions of Each LLM \

Qwen2.5-7B-Instruct:\
Qwen2.5-7B-Instruct is a powerful Chinese-English instruction-tuned large language model designed for tasks in language, \
coding, mathematics, and reasoning. As part of the Qwen2.5 series, it features enhanced knowledge, stronger coding and \
math abilities, improved instruction following, better handling of long and structured texts, and supports up to 128K \
context tokens. It also offers multilingual capabilities across over 29 languages.\

If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \
Question: {question}\n
"""


PROMPT_TEMPLATE_LLAMA = """
Answer the given question. \
Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [LLaMA-3.1-8B-Instruct]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \
    + DO NOT output <think> ... </think> as a literal string. Instead, perform your reasoning and write your thought process within these tags. That means: put your reasoning inside the tags, not as visible raw tags in the output. Only the reasoning content should appear between <think> and </think>. Do not explain or comment on the use of tags. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
You must not limit yourself to repeatedly calling a single LLM (unless its provided information is consistently the most effective and informative). \
You are encouraged to explore and utilize different LLMs to better understand their respective strengths and weaknesses. \
It is also acceptable—and recommended—to call different LLMs multiple times for the same input question to gather more comprehensive information. \


#### The Descriptions of Each LLM \

LLaMA-3.1-8B-Instruct:\
LLaMA-3.1-8B-Instruct is an 8-billion-parameter instruction-tuned language model optimized for multilingual dialogue. \
It provides strong language understanding, reasoning, and text generation performance, outperforming many open-source \
and closed-source models on standard industry benchmarks.\


If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \
Question: {question}\n
"""

PROMPT_TEMPLATE_LLAMA = """
Answer the given question. \
Use hidden internal reasoning inside <think> ... </think> to plan, verify, and decide. \
DO NOT include <think> or its contents in your output. Only output <search>, <information>, and the final <answer> as specified. \
After you receive any new <information>, update your hidden reasoning in <think> before deciding next steps. \

When you call an LLM, the response will be returned between <information> and </information>. \
If no further external knowledge is needed, provide your final answer inside <answer> ... </answer> without extra commentary. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + Put your actual answer between <answer> and </answer> using the correct format. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [LLaMA-3.1-8B-Instruct]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>. \
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \
    + The <search> tag must be exactly: <search> LLaMA-3.1-8B-Instruct:Your-Query </search>. \

USAGE GUIDANCE FOR <think> (hidden) AND <search> (visible): \
1) Decide IF you need <search>: In <think>, quickly check: \
   - Are key facts/dates/definitions uncertain? \
   - Do you need formulas, edge cases, or examples? \
   - Do you need to test or verify a hypothesis? \
   If YES, use <search>. Otherwise, proceed to <answer>. \

2) Distribute the problem into subproblems (in <think>): \
   - Facts to retrieve (definitions, constants, laws, constraints). \
   - Methods to recall (algorithms, procedures, step sequences). \
   - Assumptions to validate (units, scope, edge conditions). \
   - Calculations/examples to test (worked examples, counterexamples). \
   - Format requirements for the final answer. \
   For each uncertain subproblem, craft ONE precise <search> query. \

3) Design high-signal queries (keep them specific and actionable): \
   Prefer questions that request: short definitions, key formula(s) with variable meanings, ordered steps, minimal working examples, typical pitfalls, or concise comparisons. Avoid vague prompts. \

4) Verify and iterate: \
   - After each <information>, in <think> check for conflicts, missing pieces, or unit mismatches. \
   - If something is inconsistent or incomplete, ask a follow-up <search> with a sharper focus. \
   - Stop when you have enough to answer confidently. \

5) Manage budget and redundancy: \
   - Default to ≤ 3 <search> calls unless uncertainty remains. \
   - Do not repeat near-identical queries; vary angle (definition → example → edge case → verification). \
   - Prefer one targeted query per subproblem over a broad, catch-all query. \

6) Roles you can assign to <search> queries (pick what you need): \
   - Retriever: get definitions, constants, dates, or canonical statements. \
   - Method Summarizer: outline algorithms/procedures in 3–6 steps. \
   - Calculator-by-Example: request a minimal worked example to sanity-check. \
   - Edge-Case Scout: list common pitfalls and boundary conditions. \
   - Comparator: contrast two options, listing 3 pros/cons each. \
   - Verifier: check a candidate result, units, or assumptions. \
   - Formatter: supply a schema, template, or output structure to follow. \

7) Finalize: \
   - Produce only <answer> ... </answer> as your final output (no explanations outside the tag). \
   - Ensure it is consistent with the latest <information> and your hidden <think> checks. \

Examples of good <search> queries (use EXACT format, changing only the question text): \
   <search> LLaMA-3.1-8B-Instruct:Give a one-sentence definition of [TERM] and list its standard units. </search> \
   <search> LLaMA-3.1-8B-Instruct:State the key formula(s) for [PROBLEM], defining each variable briefly. </search> \
   <search> LLaMA-3.1-8B-Instruct:Outline a 5-step procedure to solve [TASK] from input to result. </search> \
   <search> LLaMA-3.1-8B-Instruct:Provide a minimal numeric example for [METHOD] with intermediate values. </search> \
   <search> LLaMA-3.1-8B-Instruct:List 3 common pitfalls or edge cases when solving [TOPIC]. </search> \
   <search> LLaMA-3.1-8B-Instruct:Compare [OPTION A] vs [OPTION B] in a concise list of pros and cons. </search> \
   <search> LLaMA-3.1-8B-Instruct:Check whether the assumption “[ASSUMPTION]” is valid for [CONTEXT]; answer yes/no and why briefly. </search> \
   <search> LLaMA-3.1-8B-Instruct:Suggest a counterexample that would invalidate the claim “[CLAIM].” </search> \
   <search> LLaMA-3.1-8B-Instruct:Provide recommended parameter ranges and units for [MODEL/ALGO]. </search> \
   <search> LLaMA-3.1-8B-Instruct:Return a JSON template for the final answer containing fields [FIELDS]. </search> \
   <search> LLaMA-3.1-8B-Instruct:Convert [VALUE] from [UNIT A] to [UNIT B] and show the conversion relation. </search> \
   <search> LLaMA-3.1-8B-Instruct:List the minimal necessary assumptions to solve [PROBLEM] correctly. </search> \
   <search> LLaMA-3.1-8B-Instruct:Provide a short sanity-check to validate results for [TASK]. </search> \
   <search> LLaMA-3.1-8B-Instruct:Summarize the standard notation and symbol meanings used in [FIELD/TOPIC]. </search> \

Before each LLM call, you MUST in <think> (hidden) explain: \
   + Why external information is needed. \
   + Which model is best suited (choose from [LLaMA-3.1-8B-Instruct] and use that exact name). \

Question: {question}\n
"""
