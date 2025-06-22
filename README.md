# README

This is version 1 of an AI agent designed to answer the questions on a medical
coding practice test. The agent makes use of multiple LLM's and verification strategies
to maximize accuracy.

## Usage Instructions

### Dependencies

* @langchain/langgraph: Workflow orchestration
* @langchain/openai: GPT-4o integration
* @langchain/community: PDF processing
* axios: HTTP requests for Perplexity API
  * Can consider @langchain/perplexity for python
* fs-extra: File system operations
* dotenv: Environment variable management

### Environment Variables Required 

``` env
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key
```

### File Tree

``` bash
.
├── .env
├── .gitignore
├── answers.pdf
├── answers.txt
├── package-lock.json
├── package.json
├── README.md
├── src
│   └── index.ts
├── test.pdf
└── tsconfig.json
```

### Setup

``npm install``

### Run Agent

``` bash
npm start
# Type '/run' to start the test
# Type '/status' to check progress
# Type 'quit' to exit
```

## Agent Architecture

This agent is structured with a Hierarchical Model Strategy

### Core Components

1. Primary Model: GPT-4o (OpenAI)
    * Used for question extraction and initial answering
    * Temperature: 0.1 for consistency
    * Batch processing of answering for efficiency
2. Verification Model: Perplexity AI
    * Sonar-Pro for web search
    * Sonar-Reasoning-Pro for complex analysis
    * Sonar-Deep-Research for comprehensive verification 
    * Real-time access to medical coding resources
3. Fallback System: GPT-40
    * Temperature: 0.3 for creative changes
    * Used when Perplexity fails

## Workflow Pipeline

Agent is made using **LangGraph** as their framework is incredibly robust and
enables quick development time.

``Extract -> Answer -> Verify -> Challenge -> Compare``

**Extract:**

* Process PDF test files using LangChain PDFLoader
* Extracts all 100 questions with multiple choice options
* Validates JSON structure and question numbering

**Answer:**

* Answers questions in batches of 8 for efficiency
* Provides confidence scores (1-10) for each answer
* Conservative confidence rating to identify uncertain answers
  * Could use more adjustments within the prompts
  * Could use logging to find witch are commonly uncertain problems

**Verify:**

* Targets low-confidence answers (<6) for verification
* Uses Perplexity's reasoning models with medical domain filtering
  * Used AI to generate that list, research for accuracy may be require
    for better results
* Falls back to GPT-4o if Perplexity fails
  * Ran into this at the start, but haven't since, kept the fall back
* Updates answers based on research backed evidence

**Challenge:**

* Randomly samples 50% of medium-confidence answers with scores of 6-7
* Uses Perplexity to search alternative interpretations
* Faster than full Verifications for efficiency
* Updates answers if compelling evidence found

**Compare:**

* Compares final answers against official answer key
* Generates performance metrics
* Saves results to file

## Performance & Limitations

* Final Score: The current architecture achieves a consistent score of ~70%
* Analysis: This score is a good base line and can potentially pass the test as
long as it can keep a 70%. It does reveal the limitations of relying on a
web-based agent architecture
  * It can be confidently wrong. Most likely overconfidence is found within the initial
  answers which are bypassing the verification process

## Blueprint for Higher Performance

To get past the 70% ceiling and achieve a higher score, I believe the following
architectural upgrades would be promising. However, it would require sourcing
the official PDF manuals for ICD-10-CM, CPT, and other guidelines.

1. Custom RAG with Source Documents
    * Address the limitations of web-based RAG by using proper documentation
    * Instead of using the Perplexity API I would use the source documents and
    have them chunked and stored in a efficiently accessible database as the
    core library of data. Maybe using metadata with code ranges, or organized by
    anatomical structures, or other ideas. Research more into **Vector embedding**.
2. Experts Round Table
    * Instead of simply asking the LLM's to gauge their own confidence, create 3
    distinct personas ("Doctor", "Insurance Code Auditor", "Medical Code
    Professor"), and have them each give separate answers.
    * Any disagreement found within the three answers could flag the question
    for deeper review.
3. Dispute Resolution
    * If a questioned is flagged, it can then be used with a higher end LLM
    alongside the Source Documentation to higher accuracy.
    * This mimics more of a human workflow, creating a more naturalized
    **reasoning** process for the most challenging questions.
