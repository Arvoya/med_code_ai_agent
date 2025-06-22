import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import axios from 'axios';
import * as readline from 'readline';
import * as fs from 'fs-extra';
import * as dotenv from 'dotenv';

dotenv.config();

interface Question {
  number: number;
  text: string;
  options?: string[];
  myAnswer?: string;
  confidence?: number;
  reasoning?: string;
  verifiedAnswer?: string;
  perplexityAnswer?: string;
  perplexityReasoning?: string;
  correctAnswer?: string;
  isCorrect?: boolean;
}

interface TestResults {
  totalQuestions: number;
  correctAnswers: number;
  incorrectAnswers: number;
  percentage: number;
  details: Question[];
  verifiedCount: number;
  challengedCount: number;
  perplexityCount: number;
}

const GraphState = Annotation.Root({
  extractedQuestions: Annotation<Question[]>({
    reducer: (current: Question[], update: Question[]) => update,
    default: () => []
  }),
  answeredQuestions: Annotation<Question[]>({
    reducer: (current: Question[], update: Question[]) => update,
    default: () => []
  }),
  verifiedQuestions: Annotation<Question[]>({
    reducer: (current: Question[], update: Question[]) => update,
    default: () => []
  }),
  testResults: Annotation<TestResults | null>({
    reducer: (current: TestResults | null, update: TestResults | null) => update,
    default: () => null
  })
});

type GraphStateType = typeof GraphState.State;

const llm = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0.1,
});

const llmVerify = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0.3,
});


class PerplexityAPI {
  private apiKey: string;
  private baseUrl = 'https://api.perplexity.ai/chat/completions';

  constructor() {
    this.apiKey = process.env.PERPLEXITY_API_KEY || '';
    if (!this.apiKey) {
      console.warn("⚠️  PERPLEXITY_API_KEY not found in environment variables");
    }
  }

  async query(
    prompt: string,
    systemPrompt: string = "",
    modelType: 'search' | 'reasoning' | 'research' = 'search',
    usePro: boolean = true
  ): Promise<string> {
    if (!this.apiKey) {
      throw new Error("Perplexity API key not configured");
    }

    // Choose model based on type and pro preference
    let model: string;
    let maxTokens: number;

    switch (modelType) {
      case 'search':
        model = usePro ? "sonar-pro" : "sonar";
        maxTokens = usePro ? 8000 : 4000;
        break;
      case 'reasoning':
        model = usePro ? "sonar-reasoning-pro" : "sonar-reasoning";
        maxTokens = 4000;
        break;
      case 'research':
        model = "sonar-deep-research";
        maxTokens = 6000;
        break;
      default:
        model = "sonar-pro";
        maxTokens = 8000;
    }

    try {
      const response = await axios.post(this.baseUrl, {
        model: model,
        messages: [
          ...(systemPrompt ? [{ role: "system", content: systemPrompt }] : []),
          { role: "user", content: prompt }
        ],
        max_tokens: maxTokens,
        temperature: 0.1,
        top_p: 0.9,
        return_citations: true,
        search_domain_filter: [
          "pubmed.ncbi.nlm.nih.gov",
          "cms.gov",
          "aapc.com",
          "ahima.org",
          "cdc.gov",
          "who.int",
          "icd10data.com",
          "codingclinic.com"
        ],
        return_images: false,
        return_related_questions: false,
        search_recency_filter: "month",
        top_k: 0,
        stream: false,
        presence_penalty: 0,
        frequency_penalty: 1
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      console.log(`  📡 Used Perplexity model: ${model} (${modelType})`);
      return response.data.choices[0].message.content;
    } catch (error: any) {
      if (usePro && (error.response?.status === 400 || error.response?.status === 402)) {
        console.log(`  💳 ${model} not available, falling back to standard model...`);
        return this.query(prompt, systemPrompt, modelType, false);
      }

      if (modelType === 'reasoning') {
        console.log(`  🔄 Reasoning model failed, trying search model...`);
        return this.query(prompt, systemPrompt, 'search', usePro);
      }

      console.error("Perplexity API error:", error.response?.data || error.message);
      throw new Error(`Failed to query Perplexity API: ${error.response?.data?.error || error.message}`);
    }
  }
}

const perplexity = new PerplexityAPI();

const TEST_PDF = "./test.pdf";
const ANSWERS_PDF = "./answers.pdf";
const OUTPUT_FILE = "./answers.txt";

async function processPdf(filePath: string): Promise<string> {
  try {
    console.log(`📄 Loading ${filePath}...`);

    if (!await fs.pathExists(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }

    const loader = new PDFLoader(filePath);
    const docs = await loader.load();
    const content = docs.map(doc => doc.pageContent).join('\n\n');

    console.log(`✅ PDF loaded successfully! Content length: ${content.length} characters`);
    return content;

  } catch (error) {
    console.error("❌ Error loading PDF:", error);
    throw error;
  }
}

async function extractQuestions(pdfContent: string): Promise<Question[]> {
  console.log("🔍 Extracting all medical coding questions...");

  const extractionPrompt = `Extract ALL 100 medical coding questions from this test.

Return a JSON array with ALL 100 questions:
[
  {
    "number": 1,
    "text": "Complete question text...",
    "options": ["A. First option", "B. Second option", "C. Third option", "D. Fourth option"]
  }
]

Full PDF Content:
${pdfContent}

Return ONLY the JSON array.`;

  try {
    const response = await llm.invoke([
      { role: "system", content: "Extract ALL 100 questions and return only valid JSON." },
      { role: "user", content: extractionPrompt }
    ]);

    const content = typeof response.content === 'string' ? response.content.trim() : '';

    const startIndex = content.indexOf('[');
    const endIndex = content.lastIndexOf(']');

    if (startIndex !== -1 && endIndex !== -1) {
      const jsonContent = content.substring(startIndex, endIndex + 1);
      const questions = JSON.parse(jsonContent) as Question[];
      questions.sort((a, b) => a.number - b.number);

      console.log(`✅ Extracted ${questions.length} questions`);
      return questions;
    } else {
      throw new Error("Could not find valid JSON array");
    }
  } catch (error) {
    console.error("❌ Error extracting questions:", error);
    throw error;
  }
}

async function answerQuestionsWithConfidence(questions: Question[]): Promise<Question[]> {
  console.log(`🩺 Answering ${questions.length} questions with confidence scoring...`);

  const batchSize = 8;
  const answeredQuestions: Question[] = [];

  for (let i = 0; i < questions.length; i += batchSize) {
    const batch = questions.slice(i, i + batchSize);
    console.log(`Answering batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(questions.length / batchSize)} (Questions ${batch[0].number}-${batch[batch.length - 1].number})...`);

    const batchPrompt = `You are a certified medical coding expert. Answer these ${batch.length} questions and rate your confidence CONSERVATIVELY.

IMPORTANT: Be honest about uncertainty. Medical coding has many nuances and edge cases.
- Rate 10 only if you're absolutely certain based on clear guidelines
- Rate 7-9 for solid answers with good reasoning
- Rate 4-6 if you're unsure between options
- Rate 1-3 if you're guessing

For each question:
1. Analyze each option carefully against coding guidelines
2. Consider common coding pitfalls and exceptions
3. Choose your answer (A, B, C, or D)
4. Rate your HONEST confidence (1-10)

${batch.map(q => `
Question ${q.number}: ${q.text}
${q.options ? q.options.join('\n') : ''}
`).join('\n---\n')}

Respond in this exact format:
${batch.map(q => `${q.number}: [A/B/C/D] | Confidence: [1-10]`).join('\n')}`;

    try {
      const response = await llm.invoke([
        { role: "system", content: "You are a medical coding expert. Be conservative with confidence ratings. Provide answers and confidence ratings in the exact format requested." },
        { role: "user", content: batchPrompt }
      ]);

      const content = typeof response.content === 'string' ? response.content : '';

      const answerMap = new Map<number, { answer: string, confidence: number }>();
      const lines = content.split('\n');

      for (const line of lines) {
        const match = line.match(/(\d+):\s*([A-D])\s*\|\s*Confidence:\s*(\d+)/);
        if (match) {
          const questionNum = parseInt(match[1]);
          const answer = match[2];
          const confidence = parseInt(match[3]);
          answerMap.set(questionNum, { answer, confidence });
        }
      }

      for (const question of batch) {
        const result = answerMap.get(question.number) || { answer: 'A', confidence: 5 };
        answeredQuestions.push({
          ...question,
          myAnswer: result.answer,
          confidence: result.confidence
        });
      }

    } catch (error) {
      console.error(`Error answering batch:`, error);
      for (const question of batch) {
        answeredQuestions.push({
          ...question,
          myAnswer: "A",
          confidence: 5
        });
      }
    }
  }

  const lowConfidence = answeredQuestions.filter(q => (q.confidence || 0) < 6).length;
  const mediumConfidence = answeredQuestions.filter(q => {
    const conf = q.confidence || 0;
    return conf >= 6 && conf <= 7;
  }).length;
  const highConfidence = answeredQuestions.filter(q => (q.confidence || 0) >= 9).length;
  const avgConfidence = answeredQuestions.reduce((sum, q) => sum + (q.confidence || 0), 0) / answeredQuestions.length;

  console.log(`✅ Answered ${answeredQuestions.length} questions`);
  console.log(`📊 Confidence stats: ${lowConfidence} low (<6), ${mediumConfidence} medium (6-7), ${highConfidence} high (10), avg: ${avgConfidence.toFixed(1)}`);

  return answeredQuestions;
}

async function verifyLowConfidenceAnswers(questions: Question[]): Promise<Question[]> {
  const lowConfidenceQuestions = questions.filter(q => (q.confidence || 0) < 6);

  if (lowConfidenceQuestions.length === 0) {
    console.log("✅ No low-confidence questions to verify.");
    return questions;
  }

  console.log(`🔍 Verifying ${lowConfidenceQuestions.length} low-confidence questions (Confidence < 6)...`);

  const verifiedQuestions = [...questions];
  let perplexityReasoningSuccessCount = 0;
  let perplexitySearchSuccessCount = 0;
  let gptFallbackCount = 0;

  for (let i = 0; i < lowConfidenceQuestions.length; i++) {
    const question = lowConfidenceQuestions[i];
    console.log(`Verifying ${i + 1}/${lowConfidenceQuestions.length}: Question ${question.number} (Initial Confidence: ${question.confidence})...`);

    let success = false;

    // --- Attempt 1: Perplexity Reasoning Pro ---
    try {
      const reasoningPrompt = `You are a world-class medical coding auditor. Analyze the following question, provide detailed reasoning, and give a definitive final answer.

**Question Details:**
- **Number:** ${question.number}
- **Text:** ${question.text}
- **Options:**
${question.options ? question.options.join('\n') : ''}

**Initial Assessment:**
- **Current Answer:** ${question.myAnswer}
- **Current Confidence:** ${question.confidence}/10

**Your Task:**
1.  **Analyze the Scenario:** Break down the medical scenario presented in the question.
2.  **Evaluate Each Option:** Systematically review options A, B, C, and D against the latest ICD-10-CM/CPT/HCPCS guidelines. State why each option is correct or incorrect in your reasoning.
3.  **Provide a Final Conclusion:** Based on your analysis, provide a final answer.

**Output Format:**
First, provide all your detailed reasoning as free text. Then, you MUST conclude your entire response with a single JSON code block in the following format. Do not add any text after this block.

\`\`\`json
{
  "reasoningSummary": "A brief summary of why you chose your answer and why others are incorrect.",
  "finalAnswer": "[A/B/C/D]",
  "confidence": [A number from 1-10]
}
\`\`\``;

      const response = await perplexity.query(
        reasoningPrompt,
        "You are a certified medical coding expert. Analyze the question step-by-step and provide a clear, definitive answer with reasoning, ending with the required JSON block.",
        'reasoning',
        true
      );

      const jsonBlockStart = response.lastIndexOf('```json');
      if (jsonBlockStart === -1) {
        throw new Error("Could not find the mandatory JSON block in the 'Reasoning' response.");
      }

      const jsonString = response.substring(jsonBlockStart + 7, response.lastIndexOf('```'));
      const result = JSON.parse(jsonString);

      const fullReasoning = `Perplexity Reasoning Pro Analysis:\n${response.split('```json')[0].trim()}\n\nSummary: ${result.reasoningSummary}`;

      const questionIndex = verifiedQuestions.findIndex(q => q.number === question.number);
      verifiedQuestions[questionIndex] = {
        ...verifiedQuestions[questionIndex],
        perplexityAnswer: result.finalAnswer,
        perplexityReasoning: fullReasoning,
        verifiedAnswer: result.finalAnswer,
        confidence: result.confidence,
        reasoning: fullReasoning
      };

      console.log(`  ✅ Question ${question.number}: Verified with Reasoning Pro. Answer: ${result.finalAnswer} (Conf: ${result.confidence})`);
      perplexityReasoningSuccessCount++;
      success = true;

    } catch (reasoningError: any) {
      console.log(`  ⚠️  Reasoning Pro failed for Q${question.number} (${reasoningError.message}). Trying Search Pro...`);
    }

    if (success) continue;

    // --- Attempt 2: Perplexity Search Pro (Fallback) ---
    try {
      const searchPrompt = `You are a medical coding researcher. Use your search capabilities to find the correct answer for the following question based on current guidelines.

**Question ${question.number}:** ${question.text}
${question.options ? question.options.join('\n') : ''}

**Task:**
Research the correct answer. Provide a brief justification based on your findings, then conclude with the mandatory JSON block.

**Output Format:**
You MUST conclude your entire response with a single JSON code block in the following format. Do not add any text after this block.

\`\`\`json
{
  "reasoningSummary": "A brief summary of the evidence found.",
  "finalAnswer": "[A/B/C/D]",
  "confidence": [A number from 1-10, based on the clarity of search results]
}
\`\`\``;

      const response = await perplexity.query(
        searchPrompt,
        "You are a medical coding researcher. Find the answer and respond in the required format, ending with the JSON block.",
        'search',
        true
      );

      const jsonBlockStart = response.lastIndexOf('```json');
      if (jsonBlockStart === -1) {
        throw new Error("Could not find the mandatory JSON block in the 'Search' response.");
      }

      const jsonString = response.substring(jsonBlockStart + 7, response.lastIndexOf('```'));
      const result = JSON.parse(jsonString);

      const fullReasoning = `Perplexity Search Pro Analysis:\n${response.split('```json')[0].trim()}\n\nSummary: ${result.reasoningSummary}`;

      const questionIndex = verifiedQuestions.findIndex(q => q.number === question.number);
      verifiedQuestions[questionIndex] = {
        ...verifiedQuestions[questionIndex],
        perplexityAnswer: result.finalAnswer,
        perplexityReasoning: fullReasoning,
        verifiedAnswer: result.finalAnswer,
        confidence: result.confidence,
        reasoning: fullReasoning
      };

      console.log(`  ✅ Question ${question.number}: Verified with Search Pro. Answer: ${result.finalAnswer} (Conf: ${result.confidence})`);
      perplexitySearchSuccessCount++;
      success = true;

    } catch (searchError: any) {
      console.log(`  ⚠️  Search Pro failed for Q${question.number} (${searchError.message}). Using GPT-4o fallback...`);
    }

    if (success) continue;

    // --- Attempt 3: GPT-4o (Final Fallback) ---
    try {
      const gptPrompt = `You are a senior medical coding auditor. Re-examine this question with detailed analysis and provide your response in the requested format.

**Question ${question.number}:** ${question.text}
${question.options ? question.options.join('\n') : ''}
Current Answer: ${question.myAnswer}

**Task:**
Analyze each option against current coding guidelines. Provide your reasoning, then conclude with the mandatory JSON block.

**Output Format:**
You MUST conclude your entire response with a single JSON code block in the following format. Do not add any text after this block.

\`\`\`json
{
  "reasoningSummary": "A brief summary of why your answer is correct and others are wrong.",
  "finalAnswer": "[A/B/C/D]",
  "confidence": [A number from 1-10]
}
\`\`\``;

      const gptResponse = await llmVerify.invoke(gptPrompt);
      const content = typeof gptResponse.content === 'string' ? gptResponse.content : '';

      const jsonBlockStart = content.lastIndexOf('```json');
      if (jsonBlockStart === -1) {
        throw new Error("Could not find the mandatory JSON block in the GPT-4o response.");
      }

      const jsonString = content.substring(jsonBlockStart + 7, content.lastIndexOf('```'));
      const result = JSON.parse(jsonString);

      const fullReasoning = `GPT-4o Fallback Analysis:\n${content.split('```json')[0].trim()}\n\nSummary: ${result.reasoningSummary}`;

      const questionIndex = verifiedQuestions.findIndex(q => q.number === question.number);
      verifiedQuestions[questionIndex] = {
        ...verifiedQuestions[questionIndex],
        verifiedAnswer: result.finalAnswer,
        confidence: result.confidence,
        reasoning: fullReasoning
      };

      console.log(`  ✅ Question ${question.number}: Verified with GPT-4o. Answer: ${result.finalAnswer} (Conf: ${result.confidence})`);
      gptFallbackCount++;
      success = true;

    } catch (gptError: any) {
      console.log(`  ❌ Question ${question.number}: All verification methods failed. Last error: ${gptError.message}`);
    }
  }

  const totalVerified = perplexityReasoningSuccessCount + perplexitySearchSuccessCount + gptFallbackCount;
  const changedAnswers = verifiedQuestions.filter(q => q.verifiedAnswer && q.verifiedAnswer !== q.myAnswer).length;

  console.log(`\n✅ Verification complete:`);
  console.log(`  - Total questions verified: ${totalVerified}/${lowConfidenceQuestions.length}`);
  console.log(`  - Perplexity Reasoning Pro: ${perplexityReasoningSuccessCount}`);
  console.log(`  - Perplexity Search Pro: ${perplexitySearchSuccessCount}`);
  console.log(`  - GPT-4o Fallbacks: ${gptFallbackCount}`);
  console.log(`  - Answers changed: ${changedAnswers}`);

  return verifiedQuestions;
}

async function devilsAdvocateCheck(questions: Question[]): Promise<Question[]> {
  console.log("😈 Running devil's advocate check with Perplexity on medium-confidence answers...");

  const mediumConfidenceQuestions = questions.filter(q => {
    const conf = q.confidence || 0;
    return conf >= 6 && conf <= 7;
  });

  // Select 50% of medium confidence questions randomly
  const sampleSize = Math.ceil(mediumConfidenceQuestions.length * 0.5);
  const sampledQuestions = mediumConfidenceQuestions
    .sort(() => 0.5 - Math.random())
    .slice(0, sampleSize);

  let challengedCount = 0;

  if (sampledQuestions.length > 0) {
    console.log(`🔍 Devil's advocate check with Perplexity on ${sampledQuestions.length} medium-confidence answers (50% of ${mediumConfidenceQuestions.length})...`);

    for (const question of sampledQuestions) {
      console.log(`Challenging Question ${question.number} with Perplexity...`);

      const challengePrompt = `Challenge this medical coding answer by researching potential alternative interpretations or recent guideline changes.

Question ${question.number}: ${question.text}
${question.options?.join('\n') || ''}

Current Medium-Confidence Answer: ${question.verifiedAnswer || question.myAnswer}

Please:
1. Research if there are any recent coding guideline changes that might affect this
2. Look for edge cases or exceptions that might make a different answer correct
3. Consider alternative interpretations of the clinical scenario
4. Check for common coding misconceptions

If you find compelling evidence for a different answer, provide it. Otherwise, confirm the current answer.

Format:
CHALLENGE_RESULT: [CONFIRMED/CHANGED]
FINAL_ANSWER: [A/B/C/D]
EVIDENCE: [detailed research-based reasoning]`;

      try {
        const challengeResponse = await perplexity.query(
          challengePrompt,
          "You are a critical medical coding researcher. Use latest guidelines to challenge answers.",
          'search',
          true
        );

        const resultMatch = challengeResponse.match(/CHALLENGE_RESULT:\s*(CONFIRMED|CHANGED)/);
        const answerMatch = challengeResponse.match(/FINAL_ANSWER:\s*([A-D])/);
        const evidenceMatch = challengeResponse.match(/EVIDENCE:\s*(.*?)(?=\n\n|$)/s);

        if (resultMatch && answerMatch) {
          const challengeResult = resultMatch[1];
          const finalAnswer = answerMatch[1];
          const evidence = evidenceMatch ? evidenceMatch[1].trim() : 'No evidence provided';

          if (challengeResult === 'CHANGED' && finalAnswer !== (question.verifiedAnswer || question.myAnswer)) {
            console.log(`  ⚠️  Question ${question.number}: Challenged ${question.verifiedAnswer || question.myAnswer} → ${finalAnswer} (Perplexity research)`);

            const questionIndex = questions.findIndex(q => q.number === question.number);
            if (questionIndex !== -1) {
              questions[questionIndex].verifiedAnswer = finalAnswer;
              questions[questionIndex].reasoning = `Perplexity challenge: ${evidence}`;
              challengedCount++;
            }
          } else {
            console.log(`  ✅ Question ${question.number}: Perplexity confirmed ${question.verifiedAnswer || question.myAnswer}`);
          }
        }

      } catch (error) {
        console.log(`  Error challenging question ${question.number} with Perplexity:`, error);
      }
    }
  } else {
    console.log("No medium-confidence answers to challenge");
  }

  console.log(`✅ Devil's advocate complete: ${challengedCount} answers challenged and changed by Perplexity`);

  return questions;
}

async function saveAnswersToFile(questions: Question[]): Promise<void> {
  let content = `Medical Coding Test Answers - ${new Date().toISOString()}\n`;
  content += `${'='.repeat(60)}\n`;
  content += `Total Questions: ${questions.length}\n`;

  const lowConfCount = questions.filter(q => (q.confidence || 0) < 7).length;
  const verifiedCount = questions.filter(q => q.verifiedAnswer).length;
  const changedCount = questions.filter(q => q.verifiedAnswer && q.verifiedAnswer !== q.myAnswer).length;
  const perplexityCount = questions.filter(q => q.perplexityAnswer).length;
  const challengedCount = questions.filter(q => q.reasoning?.includes("Perplexity challenge")).length;

  content += `Low Confidence (<7): ${lowConfCount}\n`;
  content += `Verified: ${verifiedCount}\n`;
  content += `Changed After Verification: ${changedCount}\n`;
  content += `Verified by Perplexity: ${perplexityCount}\n`;
  content += `Challenged by Perplexity: ${challengedCount}\n\n`;

  questions.forEach(q => {
    const finalAnswer = q.verifiedAnswer || q.myAnswer;
    content += `${q.number}. ${finalAnswer}\n`;
  });

  content += `\n${'='.repeat(60)}\nDETAILED ANSWERS:\n${'='.repeat(60)}\n\n`;

  questions.forEach(q => {
    const finalAnswer = q.verifiedAnswer || q.myAnswer;
    content += `Question ${q.number}: ${q.text}\n`;
    if (q.options) {
      q.options.forEach(option => {
        const marker = option.startsWith(finalAnswer || 'X') ? '>>> ' : '    ';
        content += `${marker}${option}\n`;
      });
    }
    content += `Initial Answer: ${q.myAnswer} (Confidence: ${q.confidence || 'N/A'})\n`;
    if (q.perplexityAnswer) {
      content += `Perplexity Answer: ${q.perplexityAnswer}\n`;
    }
    if (q.verifiedAnswer) {
      content += `Final Verified Answer: ${q.verifiedAnswer}\n`;
    }
    if (q.reasoning) {
      content += `Reasoning: ${q.reasoning}\n`;
    }
    content += `Final Answer: ${finalAnswer}\n\n`;
  });

  await fs.writeFile(OUTPUT_FILE, content, 'utf8');
  console.log(`✅ Answers saved to: ${OUTPUT_FILE}`);
}

async function compareWithAnswerKey(myAnswers: Question[], answerKeyContent: string): Promise<TestResults> {
  console.log("📊 Comparing final answers with answer key...");

  const finalAnswers = myAnswers.map(q => ({
    number: q.number,
    answer: q.verifiedAnswer || q.myAnswer
  }));

  const comparisonPrompt = `Compare my final answers with the official answer key.

My Final Answers:
${finalAnswers.map(q => `${q.number}: ${q.answer}`).join('\n')}

Official Answer Key:
${answerKeyContent}

Return JSON:
{
  "results": [
    {
      "number": 1,
      "myAnswer": "A",
      "correctAnswer": "B", 
      "isCorrect": false
    }
  ]
}`;

  try {
    const response = await llm.invoke([
      { role: "system", content: "Compare answers and return only valid JSON." },
      { role: "user", content: comparisonPrompt }
    ]);

    const content = typeof response.content === 'string' ? response.content : '';
    const jsonMatch = content.match(/\{[\s\S]*\}/);

    if (jsonMatch) {
      const comparison = JSON.parse(jsonMatch[0]);
      const results = comparison.results;

      const correctCount = results.filter((r: any) => r.isCorrect).length;
      const totalCount = results.length;
      const percentage = Math.round((correctCount / totalCount) * 100);
      const verifiedCount = myAnswers.filter(q => q.verifiedAnswer).length;
      const challengedCount = myAnswers.filter(q => q.reasoning?.includes("Perplexity challenge")).length;
      const perplexityCount = myAnswers.filter(q => q.perplexityAnswer).length;

      const testResults: TestResults = {
        totalQuestions: totalCount,
        correctAnswers: correctCount,
        incorrectAnswers: totalCount - correctCount,
        percentage: percentage,
        details: results,
        verifiedCount: verifiedCount,
        challengedCount: challengedCount,
        perplexityCount: perplexityCount
      };

      console.log(`✅ Comparison complete: ${correctCount}/${totalCount} (${percentage}%)`);
      return testResults;
    } else {
      throw new Error("Could not parse comparison results");
    }
  } catch (error) {
    console.error("❌ Error comparing answers:", error);
    throw error;
  }
}

async function extractNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  const pdfContent = await processPdf(TEST_PDF);
  const questions = await extractQuestions(pdfContent);
  return { extractedQuestions: questions };
}

async function answerNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  const answered = await answerQuestionsWithConfidence(state.extractedQuestions);
  return { answeredQuestions: answered };
}

async function verifyNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  const verified = await verifyLowConfidenceAnswers(state.answeredQuestions);
  return { verifiedQuestions: verified };
}

async function devilsAdvocateNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  const challenged = await devilsAdvocateCheck(state.verifiedQuestions);
  await saveAnswersToFile(challenged);
  return { verifiedQuestions: challenged };
}

async function compareNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  const answerKeyContent = await processPdf(ANSWERS_PDF);
  const results = await compareWithAnswerKey(state.verifiedQuestions, answerKeyContent);
  return { testResults: results };
}

const workflow = new StateGraph(GraphState)
  .addNode("extract", extractNode)
  .addNode("answer", answerNode)
  .addNode("verify", verifyNode)
  .addNode("challenge", devilsAdvocateNode)
  .addNode("compare", compareNode)
  .addEdge(START, "extract")
  .addEdge("extract", "answer")
  .addEdge("answer", "verify")
  .addEdge("verify", "challenge")
  .addEdge("challenge", "compare")
  .addEdge("compare", END);

const app = workflow.compile();

async function startCLI() {
  console.log("🏥 Advanced Medical Coding Test Assistant!");
  console.log("Features: GPT-4o + Perplexity Verification + Research-Based Devil's Advocate");
  console.log(`Files: ${TEST_PDF} → ${OUTPUT_FILE} ← ${ANSWERS_PDF}`);
  console.log("Commands: /run, /status, quit\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  let state: GraphStateType = {
    extractedQuestions: [],
    answeredQuestions: [],
    verifiedQuestions: [],
    testResults: null
  };

  const ask = () => {
    rl.question("You: ", async (input) => {
      if (input === 'quit') {
        rl.close();
        return;
      }

      if (input === '/run') {
        try {
          console.log("🚀 Starting advanced medical coding testing Agent!!!");
          const result = await app.invoke(state);
          state = { ...state, ...result };

          if (state.testResults) {
            console.log(`\n🎯 FINAL RESULTS:`);
            console.log(`Questions found: ${state.extractedQuestions.length}/100`);
            console.log(`Questions verified: ${state.testResults.verifiedCount}`);
            console.log(`Perplexity verifications: ${state.testResults.perplexityCount}`);
            console.log(`Questions challenged: ${state.testResults.challengedCount}`);
            console.log(`Score: ${state.testResults.correctAnswers}/${state.testResults.totalQuestions} (${state.testResults.percentage}%)`);
            console.log(`Answers saved to: ${OUTPUT_FILE}\n`);
          }
        } catch (error) {
          console.log("❌ Error:", error);
        }
      }
      else if (input === '/status') {
        console.log(`📋 Status:`);
        console.log(`Questions extracted: ${state.extractedQuestions.length}/100`);
        console.log(`Questions answered: ${state.answeredQuestions.length}`);
        console.log(`Questions verified: ${state.verifiedQuestions.length}`);
        console.log(`Test results: ${state.testResults ? `${state.testResults.percentage}%` : 'Not available'}\n`);
      }
      else {
        console.log("Commands: /run (start test), /status (check progress)\n");
      }

      ask();
    });
  };

  ask();
}

if (require.main === module) {
  startCLI().catch(console.error);
}
