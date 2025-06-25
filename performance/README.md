# Prompt used for Enrichment

``` js
  const prompt = `
You are a medical coding expert. Here is a code description and a real exam scenario where an AI got the question wrong.

--- CODE DESCRIPTION ---
${code} - ${originalExplanation || "(none)"}

--- QUESTION SCENARIO ---
Question: ${questionText}
Options: ${options ? options.join('\n') : "(none)"}
AI's (incorrect) reasoning: ${aiReasoning || "(none)"}

--- TASK ---
Create an accurate description for CPT code ${code} that would help an AI correctly answer this type of question.
- Explain when this code should be used, with specific emphasis on cyst excision if applicable
- Clarify common misconceptions about this code
- Distinguish this code from similar codes that might cause confusion
- Provide clear guidance on when to use this code for cyst-related procedures

Return only the improved code description.
  `.trim();
}
```
