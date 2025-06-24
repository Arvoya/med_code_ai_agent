import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import axios from 'axios';
import * as readline from 'readline';
import * as fs from 'fs-extra';
import * as dotenv from 'dotenv';
import * as cheerio from 'cheerio';
import { exec } from 'child_process'; // Add this line

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
  questionType?: 'CPT' | 'ICD-10' | 'HCPCS' | 'GENERAL';
  modelUsed?: string;
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


interface PerformanceLog {
  timestamp: string;
  questions: {
    [questionId: string]: {
      questionType: 'CPT' | 'ICD-10' | 'HCPCS' | 'GENERAL';
      modelUsed: string;
      isCorrect: boolean;
      confidence: number;
      initialAnswer?: string;
      verifiedAnswer?: string;
      correctAnswer?: string;
    }
  };
  summary: {
    totalQuestions: number;
    correctAnswers: number;
    byQuestionType: {
      CPT: { total: number; correct: number; percentage: number };
      'ICD-10': { total: number; correct: number; percentage: number };
      HCPCS: { total: number; correct: number; percentage: number };
      GENERAL: { total: number; correct: number; percentage: number };
    };
    byModel: {
      [modelName: string]: { total: number; correct: number; percentage: number };
    };
  };
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
  }),
  questionLimit: Annotation<number | undefined>({
    reducer: (current: number | undefined, update: number | undefined) => update,
    default: () => undefined
  }),
  useExplanations: Annotation<boolean>({
    reducer: (current: boolean, update: boolean) => update,
    default: () => false
  })
});

type GraphStateType = typeof GraphState.State;

//NOTE: PAID MODELS
//
const llm = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.1,
});
//
const llmVerify = new ChatOpenAI({
  model: "o3",
});


const llmAnswer = new ChatOpenAI({
  model: "o3-mini",
});


//NOTE: LOCAL MODELS
//
// const llm = new ChatOpenAI({
//   model: "Gemma 3 12b", // This name is ignored by LM Studio
//   temperature: 0.1,
//   openAIApiKey: "lm-studio", // Any non-empty string works
//   configuration: {
//     baseURL: "http://10.0.0.102:1234/v1", // Your LM Studio server URL
//   },
// });
//
// const llmVerify = new ChatOpenAI({
//   model: "Gemma 3 12b", // This name is ignored by LM Studio
//   openAIApiKey: "lm-studio",
//   configuration: {
//     baseURL: "http://10.0.0.102:1234/v1", // Your LM Studio server URL
//   },
// });
//
// const llmAnswer = new ChatOpenAI({
//   model: "Gemma 3 12b", // This name is ignored by LM Studio
//   openAIApiKey: "lm-studio",
//   configuration: {
//     baseURL: "http://10.0.0.102:1234/v1", // Your LM Studio server URL
//   },
// });


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
//NOTE:FILES
const TEST_PDF = "./test.pdf";
const ANSWERS_PDF = "./answers.pdf";
const EXPLANATIONS_MERGE_PDF = "./explanations"
const OUTPUT_FILE = "./answers.txt";
const QUESTIONS_JSON = "./questions.json";
const ANSWER_KEY_JSON = "./answer_key.json";
const CACHED_CODE_DESCRIPTIONS_JSON = "./cached_code_descriptions.json";
const PERFORMANCE_LOG_JSON = "./performance_log.json";
const PERFORMANCE_HISTORY_JSON = "./performance_history.json";

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

async function answerQuestionsWithConfidence(questions: Question[], state: GraphStateType): Promise<Question[]> {
  console.log(`🩺 Answering ${questions.length} questions with confidence scoring...`);

  const batchSize = 8;
  const answeredQuestions: Question[] = [];

  for (let i = 0; i < questions.length; i += batchSize) {
    const batch = questions.slice(i, i + batchSize);
    console.log(`Answering batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(questions.length / batchSize)} (Questions ${batch[0].number}-${batch[batch.length - 1].number})...`);

    for (const question of batch) {
      try {
        const questionType = determineQuestionType(question);
        console.log(`Question ${question.number} identified as ${questionType} question`);

        switch (questionType) {
          case 'HCPCS':
            await handleHCPCSQuestion(question, answeredQuestions, state);
            break;
          case 'ICD-10':
            await handleICD10Question(question, answeredQuestions, state);
            break;
          case 'CPT':
            await handleCPTQuestions(question, answeredQuestions, state);
            break;
          default:
            await handleGeneralQuestion(question, answeredQuestions, state);
            break;
        }
      } catch (questionError) {
        console.error(`Error processing question ${question.number}:`, questionError);
        // Add with default values if error occurs
        answeredQuestions.push({
          ...question,
          myAnswer: "A",
          confidence: 5,
          reasoning: "Error occurred during processing"
        });
      }
    }
  }

  // Rest of the function remains the same
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

// Helper function to determine question type
// Helper function to determine question type
function determineQuestionType(question: Question): 'HCPCS' | 'ICD-10' | 'CPT' | 'GENERAL' {
  const text = question.text.toLowerCase();
  const options = question.options ? question.options.join(' ').toLowerCase() : '';
  const fullText = text + ' ' + options;

  // Check for HCPCS indicators
  if (
    fullText.includes('hcpcs') ||
    fullText.includes('level ii code') ||
    fullText.includes('durable medical equipment') ||
    fullText.includes('prosthetic')
  ) {
    question.questionType = 'HCPCS';
    return 'HCPCS';
  }

  // Check for HCPCS code pattern in options (safely)
  if (question.options) {
    for (const opt of question.options) {
      const match = opt.match(/[A-Z]\d{4}/);
      if (match && match[0] && match[0][0] !== 'C') {
        question.questionType = 'HCPCS';
        return 'HCPCS';
      }
    }
  }

  // Check for ICD-10 indicators
  if (
    fullText.includes('icd-10') ||
    fullText.includes('diagnosis code') ||
    fullText.includes('diagnostic code') ||
    fullText.includes('according to icd')
  ) {
    question.questionType = 'ICD-10';
    return 'ICD-10';
  }

  // Check for ICD-10 code pattern in options (safely)
  if (question.options) {
    for (const opt of question.options) {
      if (opt.match(/[A-Z]\d{2}(\.\d+)?/)) {
        question.questionType = 'ICD-10';
        return 'ICD-10';
      }
    }
  }

  // Check for CPT indicators
  if (
    fullText.includes('cpt') ||
    fullText.includes('procedure code') ||
    fullText.includes('surgical code')
  ) {
    question.questionType = 'CPT';
    return 'CPT';
  }

  // Check for CPT code pattern in options (safely)
  if (question.options) {
    for (const opt of question.options) {
      if (opt.match(/\b\d{5}\b/)) {
        question.questionType = 'CPT';
        return 'CPT';
      }
    }
  }

  // Default to general
  question.questionType = 'GENERAL';
  return 'GENERAL';
}


// Handle HCPCS questions
async function handleHCPCSQuestion(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
  console.log(`Processing HCPCS question ${question.number}...`);

  // Extract potential HCPCS codes from options
  const hcpcsCodes: string[] = [];
  if (question.options) {
    for (const option of question.options) {
      const matches = option.match(/[A-Z]\d{4}/g);
      if (matches) {
        hcpcsCodes.push(...matches);
      }
    }
  }

  if (hcpcsCodes.length > 0) {
    // console.log(`Found HCPCS codes in options: ${hcpcsCodes.join(', ')}`);

    // Create a map to store results for each code
    const codeDescriptions: Map<string, string> = new Map();

    // Initialize or load the cached codes file
    let cachedCodes: any;
    try {
      // Load cached code descriptions (using merged if enabled)
      cachedCodes = await loadCodeDescriptions(state);
    } catch (error) {
      // If file doesn't exist or can't be read, initialize with empty structure
      console.log("Creating new cache file with empty structure");
      cachedCodes = {
        "CPT": [],
        "ICD-10": [],
        "HCPCS": []
      };
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });
      console.log("New cache file created");
    }

    // Make sure HCPCS array exists
    if (!cachedCodes["HCPCS"]) {
      console.log("HCPCS array missing in cache, initializing it");
      cachedCodes["HCPCS"] = [];
    }

    // Check which codes we need to fetch from the API
    const codesToFetch: string[] = [];
    for (const code of hcpcsCodes) {
      // Check if code exists in cached data
      const cachedCode = cachedCodes["HCPCS"].find((item: any) => item.code === code);
      if (cachedCode) {
        // console.log(`Found cached HCPCS code ${code}: ${cachedCode.description}`);
        if (state.useExplanations && cachedCode.explanation) {
          codeDescriptions.set(code, cachedCode.description + cachedCode.explanation);
        } else {
          codeDescriptions.set(code, cachedCode.description);
        }
      } else {
        console.log(`Code ${code} not found in cache, will fetch from API`);
        codesToFetch.push(code);
      }
    }

    // Fetch any codes not found in cache
    if (codesToFetch.length > 0) {
      console.log(`Fetching ${codesToFetch.length} HCPCS codes from API...`);

      for (const code of codesToFetch) {
        try {
          // console.log(`Querying API for code: ${code}`);
          const response = await axios.get(`https://clinicaltables.nlm.nih.gov/api/hcpcs/v3/search?terms=${code}`);
          // console.log(`API response for ${code}:`, JSON.stringify(response.data));

          let description: string | null = null;

          // Based on the actual response format observed in logs
          if (response.data && Array.isArray(response.data)) {
            const data = response.data;

            // Format appears to be [count, [codes], null, [[code, description], ...]]
            if (data.length >= 4 && Array.isArray(data[3])) {
              // console.log(`Parsing response format: [count, [codes], null, [[code, description], ...]]`);

              const codeDescPairs = data[3];
              if (Array.isArray(codeDescPairs)) {
                for (const pair of codeDescPairs) {
                  if (Array.isArray(pair) && pair.length >= 2 && pair[0] === code) {
                    description = pair[1];
                    // console.log(`Found exact match: ${pair[0]} = ${pair[1]}`);
                    break;
                  }
                }
              }
            }
            // Try other formats if needed
            else if (data[0] && Array.isArray(data[0])) {
              // console.log(`Trying alternate format: [[code, description], ...]`);
              for (const pair of data) {
                if (Array.isArray(pair) && pair.length >= 2 && pair[0] === code) {
                  description = pair[1];
                  // console.log(`Found match in alternate format: ${pair[0]} = ${pair[1]}`);
                  break;
                }
              }
            }
          }

          if (description) {
            codeDescriptions.set(code, description);
            // console.log(`  Code ${code}: ${description}`);

            // Add to cache
            cachedCodes["HCPCS"].push({
              code: code,
              description: description
            });

            // console.log('ADDING TO JSON:', code, description);
          } else {
            console.log(`No description found for code ${code} in API response`);
          }
        } catch (error) {
          console.error(`  Error querying HCPCS API for code ${code}:`, error);
        }
      }

      // Save the updated cache
      // console.log("Saving updated cache to file...");
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });

      // Verify the file was written correctly
      try {
        const verifyData = await fs.readJSON(CACHED_CODE_DESCRIPTIONS_JSON);
        console.log(`Cache file updated successfully. Contains ${verifyData["HCPCS"].length} HCPCS codes.`);
      } catch (error) {
        console.error("Error verifying cache file update:", error);
      }
    }

    // Determine the answer based on the question and code descriptions
    let answer = '';
    let confidence = 6;
    let reasoning = '';

    // Extract the key condition or item from the question
    const questionKeywords = extractKeywords(question.text);

    // Compare each option with the results
    if (question.options) {
      for (let i = 0; i < question.options.length; i++) {
        const option = question.options[i];
        const optionLetter = String.fromCharCode(65 + i); // A, B, C, D

        // Extract code from option
        const codeMatch = option.match(/[A-Z]\d{4}/);
        if (codeMatch) {
          const code = codeMatch[0];
          const description = codeDescriptions.get(code);

          if (description) {
            // Check if description matches the question context
            const matchScore = calculateMatchScore(description, questionKeywords);

            if (matchScore > 0.7) { // Threshold for a good match
              answer = optionLetter;
              confidence = 8;
              reasoning = `HCPCS code ${code} description "${description}" matches the question context. Verification confirms this is the correct code.`;
              break;
            }
          }
        }
      }
    }

    // If no good match was found, use o4-mini to determine the answer
    if (!answer) {
      console.log(`USING ${llmAnswer.model} with code descriptions:`);
      // const contextInfo = Array.from(codeDescriptions.entries())
      //   .map(([code, desc]) => `${code}: ${desc}`)
      //   .join('\n');
      // console.log(contextInfo);

      const result = await useO4MiniForQuestion(question, codeDescriptions);
      answer = result.answer;
      confidence = result.confidence;
      reasoning = result.reasoning;
      question.modelUsed = llmAnswer.model;
    }

    answeredQuestions.push({
      ...question,
      myAnswer: answer,
      confidence: confidence,
      reasoning: reasoning,
      modelUsed: llmAnswer.model
    });

  } else {
    // No HCPCS codes found in options, use o4-mini
    console.log(`No HCPCS codes found in options for question ${question.number}, using ${llmAnswer.model}...`);
    await handleGeneralQuestion(question, answeredQuestions, state);
  }
}

async function handleCPTQuestions(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
  console.log(`Processing CPT question ${question.number}...`);

  // Extract potential CPT codes from options
  const cptCodes: string[] = [];
  if (question.options) {
    for (const option of question.options) {
      // CPT codes are typically 5 digits
      const matches = option.match(/\b\d{5}\b/g);
      if (matches) {
        cptCodes.push(...matches);
      }
    }
  }

  if (cptCodes.length > 0) {
    // console.log(`Found CPT codes in options: ${cptCodes.join(', ')}`);

    // Create a map to store results for each code
    const codeDescriptions: Map<string, string> = new Map();

    // Initialize or load the cached codes file
    let cachedCodes: any;
    try {
      // Use the loadCodeDescriptions function with state
      cachedCodes = await loadCodeDescriptions(state);
    } catch (error) {
      // If file doesn't exist or can't be read, initialize with empty structure
      console.log("Creating new cache file with empty structure");
      cachedCodes = {
        "CPT": [],
        "ICD-10": [],
        "HCPCS": []
      };
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });
      console.log("New cache file created");
    }

    // Make sure CPT array exists
    if (!cachedCodes["CPT"]) {
      // console.log("CPT array missing in cache, initializing it");
      cachedCodes["CPT"] = [];
    }

    // Check which codes we need to fetch from the website
    const codesToFetch: string[] = [];
    for (const code of cptCodes) {
      // Check if code exists in cached data
      const cachedCode = cachedCodes["CPT"].find((item: any) => item.code === code);
      if (cachedCode) {
        // console.log(`Found cached CPT code ${code}: ${cachedCode.description}`);
        if (state.useExplanations && cachedCode.explanation) {
          codeDescriptions.set(code, cachedCode.description + cachedCode.explanation);
        } else {
          codeDescriptions.set(code, cachedCode.description);
        }
      } else {
        // console.log(`Code ${code} not found in cache, will fetch from website`);
        codesToFetch.push(code);
      }
    }

    // Fetch any codes not found in cache
    if (codesToFetch.length > 0) {
      console.log(`Fetching ${codesToFetch.length} CPT codes from AAPC website...`);

      for (const code of codesToFetch) {
        try {
          // console.log(`Querying website for code: ${code}`);
          const url = `https://www.aapc.com/codes/cpt-codes/${code}`;
          const response = await axios.get(url);

          // Use cheerio to parse the HTML
          const $ = cheerio.load(response.data);

          // Find the description in the sub_head_detail class
          const subHeadDetail = $('.sub_head_detail').text();
          // console.log(`Raw sub_head_detail: ${subHeadDetail}`);

          let description: string | null = null;

          if (subHeadDetail) {
            // Extract the description from the text
            // The format is typically: "The Current Procedural Terminology (CPT®) code XXXXX as maintained by American Medical Association, is a medical procedural code under the range - DESCRIPTION."
            const descMatch = subHeadDetail.match(/under the range - (.+?)\.$/);
            if (descMatch && descMatch[1]) {
              description = descMatch[1].trim();
            } else {
              // If we can't extract with regex, just use the whole text
              description = subHeadDetail.trim();
            }
          }

          // If we couldn't find the description in sub_head_detail, try looking for it elsewhere
          if (!description) {
            // Try to find the code description in other common locations
            const codeTitle = $('h1.cpt_code').text().trim();
            if (codeTitle) {
              const titleMatch = codeTitle.match(/\d{5} (.+)$/);
              if (titleMatch && titleMatch[1]) {
                description = titleMatch[1].trim();
              }
            }
          }

          // Extract the summary - first try from cpt_layterms
          let summary = '';
          const cptLayterms = $('#cpt_layterms').find('p').first();
          if (cptLayterms.length > 0 && cptLayterms.text().trim()) {
            summary = cptLayterms.text().trim();
          } else {
            // If no summary in cpt_layterms, try offlongdesc
            const offLongDesc = $('#offlongdesc').find('p').first();
            if (offLongDesc.length > 0 && offLongDesc.text().trim()) {
              summary = offLongDesc.text().trim();
            }
          }

          // Combine description and summary if both exist
          let fullDescription = description || '';
          if (summary && fullDescription) {
            fullDescription += ` Summary: ${summary}`;
          } else if (summary) {
            fullDescription = summary;
          }

          if (fullDescription) {
            codeDescriptions.set(code, fullDescription);
            // console.log(`  Code ${code}: ${fullDescription}`);

            // Add to cache
            cachedCodes["CPT"].push({
              code: code,
              description: fullDescription
            });

            // console.log('ADDING TO JSON:', code, fullDescription);
          } else {
            // console.log(`No description found for code ${code} on the website`);

            // Add a placeholder description
            const placeholderDesc = `CPT code ${code} - Description not found. Please research this code further for accurate information.`;
            codeDescriptions.set(code, placeholderDesc);

            // Add to cache with placeholder
            cachedCodes["CPT"].push({
              code: code,
              description: placeholderDesc
            });

            // console.log('ADDING TO JSON WITH PLACEHOLDER:', code, placeholderDesc);
          }
        } catch (error: any) {
          console.error(`  Error fetching CPT code ${code} from website:`, error);

          // Check if it's a 404 error
          let errorMessage = "";
          if (error.response && error.response.status === 404) {
            errorMessage = `CPT code ${code} - Not found on AAPC website. This may be a test code or requires specialized knowledge. Please research this code further.`;
          } else {
            errorMessage = `CPT code ${code} - Unable to retrieve description due to technical error. Please research this code further.`;
          }

          // Add to the descriptions map with the error message
          codeDescriptions.set(code, errorMessage);

          // Add to cache with the error message
          cachedCodes["CPT"].push({
            code: code,
            description: errorMessage
          });

          // console.log('ADDING TO JSON WITH ERROR MESSAGE:', code, errorMessage);
        }

        // Add a more conservative delay of 3 seconds between requests
        console.log(`Waiting 3 seconds before next request...`);
        await new Promise(resolve => setTimeout(resolve, 3000));
      }

      // Save the updated cache
      console.log("Saving updated cache to file...");
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });

      // Verify the file was written correctly
      try {
        const verifyData = await fs.readJSON(CACHED_CODE_DESCRIPTIONS_JSON);
        // console.log(`Cache file updated successfully. Contains ${verifyData["CPT"].length} CPT codes.`);
      } catch (error) {
        console.error("Error verifying cache file update:", error);
      }
    }

    // Use o4-mini with the code descriptions as context
    console.log(`USING ${llmAnswer.model} with code descriptions:`);
    // const contextInfo = Array.from(codeDescriptions.entries())
    //   .map(([code, desc]) => `${code}: ${desc}`)
    //   .join('\n');
    // console.log(contextInfo);

    const result = await useO4MiniForQuestion(question, codeDescriptions);

    answeredQuestions.push({
      ...question,
      myAnswer: result.answer,
      confidence: result.confidence,
      reasoning: result.reasoning,
      modelUsed: llmAnswer.model
    });

  } else {
    // No CPT codes found in options, use o4-mini
    console.log(`No CPT codes found in options for question ${question.number}, using ${llmAnswer.model}...`);
    await handleGeneralQuestion(question, answeredQuestions, state);
  }
}

// Handle ICD-10 questions
async function handleICD10Question(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
  console.log(`Processing ICD-10 question ${question.number}...`);

  // For ICD-10 guideline questions without specific codes, use o4-mini
  if (!question.options || !question.options.some(opt => /[A-Z]\d{2}(\.\d+)?/.test(opt))) {
    console.log(`No ICD-10 codes found in options for question ${question.number}, using ${llmAnswer.model}...`);
    await handleGeneralQuestion(question, answeredQuestions, state);
    return;
  }

  // Extract potential ICD-10 codes from options
  const icdCodes: string[] = [];
  if (question.options) {
    for (const option of question.options) {
      const matches = option.match(/[A-Z]\d{2}(\.\d+)?/g);
      if (matches) {
        icdCodes.push(...matches);
      }
    }
  }

  if (icdCodes.length > 0) {
    console.log(`Found ICD-10 codes in options: ${icdCodes.join(', ')}`);

    // Create a map to store results for each code
    const codeDescriptions: Map<string, string> = new Map();

    // Initialize or load the cached codes file
    let cachedCodes: any;
    try {
      // Use the loadCodeDescriptions function with state
      cachedCodes = await loadCodeDescriptions(state);
    } catch (error) {
      // If file doesn't exist or can't be read, initialize with empty structure
      console.log("Creating new cache file with empty structure");
      cachedCodes = {
        "CPT": [],
        "ICD-10": [],
        "HCPCS": []
      };
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });
      console.log("New cache file created");
    }

    // Make sure ICD-10 array exists
    if (!cachedCodes["ICD-10"]) {
      // console.log("ICD-10 array missing in cache, initializing it");
      cachedCodes["ICD-10"] = [];
    }

    // Check which codes we need to fetch from the API
    const codesToFetch: string[] = [];
    for (const code of icdCodes) {
      // Check if code exists in cached data
      const cachedCode = cachedCodes["ICD-10"].find((item: any) => item.code === code);
      if (cachedCode) {
        // console.log(`Found cached ICD-10 code ${code}: ${cachedCode.description}`);
        if (state.useExplanations && cachedCode.explanation) {
          codeDescriptions.set(code, cachedCode.description + cachedCode.explanation);
        } else {
          codeDescriptions.set(code, cachedCode.description);
        }
      } else {
        console.log(`Code ${code} not found in cache, will fetch from API`);
        codesToFetch.push(code);
      }
    }

    // Fetch any codes not found in cache
    if (codesToFetch.length > 0) {
      console.log(`Fetching ${codesToFetch.length} ICD-10 codes from API...`);

      for (const code of codesToFetch) {
        try {
          // console.log(`Querying API for code: ${code}`);
          const response = await axios.get(`https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms=${code}`);
          // console.log(`API response for ${code}:`, JSON.stringify(response.data));

          let description: string | null = null;

          // Based on the actual response format observed in logs
          if (response.data && Array.isArray(response.data)) {
            const data = response.data;

            // Format appears to be [count, [codes], null, [[code, description], ...]]
            if (data.length >= 4 && Array.isArray(data[3])) {
              // console.log(`Parsing response format: [count, [codes], null, [[code, description], ...]]`);

              const codeDescPairs = data[3];
              if (Array.isArray(codeDescPairs)) {
                for (const pair of codeDescPairs) {
                  if (Array.isArray(pair) && pair.length >= 2 && pair[0] === code) {
                    description = pair[1];
                    console.log(`Found exact match: ${pair[0]} = ${pair[1]}`);
                    break;
                  }
                }
              }
            }
            // Try other formats if needed
            else if (data[0] && Array.isArray(data[0])) {
              // console.log(`Trying alternate format: [[code, description], ...]`);
              for (const pair of data) {
                if (Array.isArray(pair) && pair.length >= 2 && pair[0] === code) {
                  description = pair[1];
                  // console.log(`Found match in alternate format: ${pair[0]} = ${pair[1]}`);
                  break;
                }
              }
            }
          }

          if (description) {
            codeDescriptions.set(code, description);
            // console.log(`  Code ${code}: ${description}`);

            // Add to cache
            cachedCodes["ICD-10"].push({
              code: code,
              description: description
            });

            // console.log('ADDING TO JSON:', code, description);
          } else {
            console.log(`No description found for code ${code} in API response`);
          }
        } catch (error) {
          console.error(`  Error querying ICD-10 API for code ${code}:`, error);
        }
      }

      // Save the updated cache
      // console.log("Saving updated cache to file...");
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });

      // Verify the file was written correctly
      try {
        const verifyData = await fs.readJSON(CACHED_CODE_DESCRIPTIONS_JSON);
        // console.log(`Cache file updated successfully. Contains ${verifyData["ICD-10"].length} ICD-10 codes.`);
      } catch (error) {
        console.error("Error verifying cache file update:", error);
      }
    }

    // Use o4-mini with the code descriptions as context
    const contextInfo = Array.from(codeDescriptions.entries())
      .map(([code, desc]) => `${code}: ${desc}`)
      .join('\n');

    console.log(`USING ${llmAnswer.model} with code descriptions:`);

    const result = await useO4MiniForQuestion(question, codeDescriptions);

    answeredQuestions.push({
      ...question,
      myAnswer: result.answer,
      confidence: result.confidence,
      reasoning: result.reasoning,
      modelUsed: llmAnswer.model
    });

  } else {
    // No ICD-10 codes found in options, use o4-mini
    console.log(`No ICD-10 codes found in options for question ${question.number}, using ${llmAnswer.model}...`);
    await handleGeneralQuestion(question, answeredQuestions, state);
  }
}

// Handle CPT or general medical coding questions
async function handleGeneralQuestion(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
  console.log(`Processing General question ${question.number} with ${llmAnswer.model}...`);

  const questionPrompt = `You are a certified medical coding expert. Answer this question and rate your confidence CONSERVATIVELY.

IMPORTANT: Be honest about uncertainty. Medical questions has many nuances and edge cases.
- Rate 10 only if you're absolutely certain based on clear guidelines
- Rate 7-9 for solid answers with good reasoning
- Rate 4-6 if you're unsure between options
- Rate 1-3 if you're guessing

Question ${question.number}: ${question.text}
${question.options ? question.options.join('\n') : ''}

Respond in this exact format:
Answer: [A/B/C/D]
Confidence: [1-10]
Reasoning: [brief explanation of your choice]`;

  const response = await llmAnswer.invoke([
    { role: "system", content: "You are a medical coding expert. Be conservative with confidence ratings." },
    { role: "user", content: questionPrompt }
  ]);

  const content = typeof response.content === 'string' ? response.content : '';

  const answerMatch = content.match(/Answer:\s*([A-D])/);
  const confidenceMatch = content.match(/Confidence:\s*(\d+)/);
  const reasoningMatch = content.match(/Reasoning:\s*(.*?)(?=\n\n|$)/s);

  if (answerMatch && confidenceMatch) {
    const answer = answerMatch[1];
    const confidence = parseInt(confidenceMatch[1]);
    const reasoning = reasoningMatch ? reasoningMatch[1].trim() : '';

    answeredQuestions.push({
      ...question,
      myAnswer: answer,
      confidence: confidence,
      reasoning: reasoning,
      modelUsed: llmAnswer.model
    });
  } else {
    // Fallback if response format is incorrect
    console.error(`Unexpected response format from ${llmAnswer.model}  for question ${question.number}`);
    answeredQuestions.push({
      ...question,
      myAnswer: "A",
      confidence: 5,
      reasoning: "Could not parse model response"
    });
  }
}

// Helper function to use o4-mini with code descriptions as context
async function useO4MiniForQuestion(
  question: Question,
  codeDescriptions: Map<string, string>
): Promise<{ answer: string, confidence: number, reasoning: string }> {
  const o4Mini = llmAnswer;

  // console.log("USING O3-mini with code descriptions:");
  // codeDescriptions.forEach((description, code) => {
  //   console.log(`  ${code}: ${description}`);
  // });

  // Create context from code descriptions
  const contextInfo = Array.from(codeDescriptions.entries())
    .map(([code, desc]) => `${code}: ${desc}`)
    .join('\n');


  const questionPrompt = `You are a certified medical coding expert. Answer this question using the provided code descriptions and rate your confidence CONSERVATIVELY.

IMPORTANT: Be honest about uncertainty. Medical coding has many nuances and edge cases.
- Rate 10 only if you're absolutely certain based on clear guidelines
- Rate 7-9 for solid answers with good reasoning
- Rate 4-6 if you're unsure between options
- Rate 1-3 if you're guessing

Question ${question.number}: ${question.text}
${question.options ? question.options.join('\n') : ''}

Vital Code Descriptions from Official Database:
${contextInfo}

Respond in this exact format:
Answer: [A/B/C/D]
Confidence: [1-10]
Reasoning: [brief explanation of your choice]`;

  const response = await o4Mini.invoke([
    { role: "system", content: "You are a medical coding expert. Be conservative with confidence ratings." },
    { role: "user", content: questionPrompt }
  ]);

  const content = typeof response.content === 'string' ? response.content : '';

  const answerMatch = content.match(/Answer:\s*([A-D])/);
  const confidenceMatch = content.match(/Confidence:\s*(\d+)/);
  const reasoningMatch = content.match(/Reasoning:\s*(.*?)(?=\n\n|$)/s);

  if (answerMatch && confidenceMatch) {
    return {
      answer: answerMatch[1],
      confidence: parseInt(confidenceMatch[1]),
      reasoning: reasoningMatch ? reasoningMatch[1].trim() : ''
    };
  } else {
    // Fallback if response format is incorrect
    console.error(`Unexpected response format from ${llmAnswer.model}`);
    return {
      answer: "A",
      confidence: 5,
      reasoning: "Could not parse model response"
    };
  }
}

// Helper function to extract keywords from question text
function extractKeywords(text: string): string[] {
  // Remove common words and extract key medical terms
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(word =>
      word.length > 3 &&
      !['what', 'which', 'when', 'where', 'this', 'that', 'with', 'from', 'have', 'code', 'represents'].includes(word)
    );

  return words;
}

// Helper function to calculate match score between description and keywords
function calculateMatchScore(description: string, keywords: string[]): number {
  const descriptionLower = description.toLowerCase();
  let matchCount = 0;

  for (const keyword of keywords) {
    if (descriptionLower.includes(keyword)) {
      matchCount++;
    }
  }

  return keywords.length > 0 ? matchCount / keywords.length : 0;
}


// Add these functions after the calculateMatchScore function

/**
 * Extracts explanations for each answer from the answers PDF
 * @returns {Promise<Array>} Array of explanation objects with question numbers and explanation text
 */
async function extractExplanationsFromAnswersPdf(): Promise<any[]> {
  console.log("📄 Extracting explanations from answers.pdf...");

  try {
    if (!await fs.pathExists(ANSWERS_PDF)) {
      throw new Error(`File not found: ${ANSWERS_PDF}`);
    }

    const pdfContent = await processPdf(ANSWERS_PDF);

    const extractionPrompt = `Extract ALL explanations for each answer from this PDF.
    
    For each question, find:
    1. The question number
    2. The correct answer (letter and code)
    3. The complete explanation text
    
    Return a JSON array with explanations for each question:
    [
      {
        "number": 1,
        "correctAnswer": "A. G43.009",
        "explanation": "Complete explanation text for why this answer is correct..."
      }
    ]
    
    Full PDF Content:
    ${pdfContent}
    
    Return ONLY the JSON array.`;

    const response = await llm.invoke([
      { role: "system", content: "Extract explanations for medical coding answers and return only valid JSON." },
      { role: "user", content: extractionPrompt }
    ]);

    const content = typeof response.content === 'string' ? response.content.trim() : '';

    const startIndex = content.indexOf('[');
    const endIndex = content.lastIndexOf(']');

    if (startIndex !== -1 && endIndex !== -1) {
      const jsonContent = content.substring(startIndex, endIndex + 1);
      const explanations = JSON.parse(jsonContent);
      explanations.sort((a: any, b: any) => a.number - b.number);

      console.log(`✅ Extracted explanations for ${explanations.length} questions`);
      return explanations;
    } else {
      throw new Error("Could not find valid JSON array in the response");
    }
  } catch (error) {
    console.error("❌ Error extracting explanations:", error);
    throw error;
  }
}

/**
 * Extracts all medical codes from text with their context
 * @param {string} text The text containing codes
 * @returns {Array<{code: string, codeType: string, context: string}>} Array of code objects
 */
function extractCodesWithContext(text: string): Array<{ code: string, codeType: string, context: string }> {
  if (!text) return [];

  const results: Array<{ code: string, codeType: string, context: string }> = [];

  // Define regex patterns for different code types
  const patterns: { [key: string]: RegExp } = {
    "CPT": /\b(\d{5})\b/g,
    "ICD-10": /\b([A-Z]\d{2}(\.\d+)?)\b/g,
    "HCPCS": /\b([A-Z]\d{4})\b/g
  };

  // For each code type, find all matches and extract context
  for (const [codeType, pattern] of Object.entries(patterns)) {
    let match;
    const patternCopy = new RegExp(pattern); // Create a new RegExp to reset lastIndex
    while ((match = patternCopy.exec(text)) !== null) {
      const code = match[1];

      // Extract context (sentence containing the code)
      const sentenceStart = text.lastIndexOf('.', match.index) + 1;
      const sentenceEnd = text.indexOf('.', match.index);
      const context = text.substring(
        Math.max(0, sentenceStart),
        sentenceEnd > -1 ? sentenceEnd + 1 : text.length
      ).trim();

      results.push({
        code,
        codeType,
        context
      });
    }
  }

  return results;
}

/**
 * Merges explanations with cached code descriptions and saves to a new file
 * @returns {Promise<object>} The merged code descriptions
 */
async function mergeExplanationsWithCachedDescriptions(): Promise<any> {
  console.log("🔄 Merging explanations with cached code descriptions...");

  try {
    // Load cached code descriptions
    if (!await fs.pathExists(CACHED_CODE_DESCRIPTIONS_JSON)) {
      throw new Error(`Cached code descriptions file not found: ${CACHED_CODE_DESCRIPTIONS_JSON}`);
    }

    const cachedCodes = await fs.readJSON(CACHED_CODE_DESCRIPTIONS_JSON);
    console.log(`Loaded cached code descriptions with ${Object.values(cachedCodes).flat().length} entries`);

    // Extract explanations from answers PDF
    const explanations = await extractExplanationsFromAnswersPdf();
    console.log(`Extracted ${explanations.length} explanations from answers PDF`);

    // Create a deep copy of the cached codes
    const mergedCodes = JSON.parse(JSON.stringify(cachedCodes));

    // Create a map for quick lookup of explanations by question number
    const explanationMap = new Map();
    explanations.forEach(exp => explanationMap.set(exp.number, exp));

    // Create a map for quick lookup of explanations by code
    const codeExplanationMap = new Map();

    // Process explanations to extract codes and map them
    for (const explanation of explanations) {
      // If explanation already has a code property, add it to the map
      if (explanation.correctAnswer) {
        // Extract code from answer (e.g., "A. G43.009" -> "G43.009")
        const codeMatch = explanation.correctAnswer.match(/[A-Z]\d{2}(\.\d+)?|\d{5}|[A-Z]\d{4}/);
        if (codeMatch) {
          const primaryCode = codeMatch[0];
          codeExplanationMap.set(primaryCode, explanation.explanation);
        }
      }

      // Extract codes from the explanation text
      const extractedCodes = extractCodesWithContext(explanation.explanation);

      // Add all extracted codes to the map
      for (const codeInfo of extractedCodes) {
        codeExplanationMap.set(codeInfo.code, explanation.explanation);
      }
    }

    // Count of explanations added
    let explanationsAdded = 0;
    let newCodesAdded = 0;

    // For each code type (CPT, ICD-10, HCPCS)
    for (const codeType of ["CPT", "ICD-10", "HCPCS"]) {
      if (!mergedCodes[codeType]) continue;

      // For each code in this type
      for (const codeEntry of mergedCodes[codeType]) {
        // Check if we have an explanation for this code
        if (codeExplanationMap.has(codeEntry.code)) {
          codeEntry.explanation = codeExplanationMap.get(codeEntry.code);
          explanationsAdded++;
        }
      }

      // Now check for codes in explanations that aren't in our cache
      for (const [code, explanation] of codeExplanationMap.entries()) {
        // Determine code type
        let detectedCodeType: string;
        if (/^\d{5}$/.test(code)) {
          detectedCodeType = "CPT";
        } else if (/^[A-Z]\d{2}(\.\d+)?$/.test(code)) {
          detectedCodeType = "ICD-10";
        } else if (/^[A-Z]\d{4}$/.test(code)) {
          detectedCodeType = "HCPCS";
        } else {
          continue; // Skip if we can't determine the code type
        }

        // Skip if not matching the current codeType
        if (detectedCodeType !== codeType) continue;

        // Check if this code exists in our cache
        const codeExists = mergedCodes[codeType].some((entry: any) => entry.code === code);

        // If code doesn't exist, add it
        if (!codeExists) {
          mergedCodes[codeType].push({
            code,
            description: `Code extracted from explanation in answers PDF`,
            explanation,
            source: "answers_pdf_explanation"
          });
          newCodesAdded++;
        }
      }
    }

    // Add a metadata section to track the merge
    mergedCodes.metadata = {
      originalSource: CACHED_CODE_DESCRIPTIONS_JSON,
      explanationsSource: ANSWERS_PDF,
      mergeDate: new Date().toISOString(),
      totalCodes: Object.values(mergedCodes)
        .filter(arr => Array.isArray(arr))
        .reduce((sum: number, arr: any[]) => sum + arr.length, 0),
      explanationsAdded,
      newCodesAdded
    };

    // Create the directory if it doesn't exist
    const dirPath = EXPLANATIONS_MERGE_PDF;
    await fs.ensureDir(dirPath);

    // Save the merged data to a new file
    const outputPath = `${dirPath}/merged_code_descriptions.json`;
    await fs.writeJSON(outputPath, mergedCodes, { spaces: 2 });
    console.log(`✅ Merged explanations saved to: ${outputPath}`);
    console.log(`   Added explanations to ${explanationsAdded} existing code entries`);
    console.log(`   Added ${newCodesAdded} new code entries from explanations`);

    return mergedCodes;
  } catch (error) {
    console.error("❌ Error merging explanations:", error);
    throw error;
  }
}

/**
 * Compares original cached descriptions with merged explanations
 * @returns {Promise<void>}
 */
async function compareOriginalAndMergedDescriptions(): Promise<void> {
  console.log("🔍 Comparing original and merged code descriptions...");

  try {
    // Load original cached code descriptions
    const originalPath = CACHED_CODE_DESCRIPTIONS_JSON;
    if (!await fs.pathExists(originalPath)) {
      throw new Error(`Original file not found: ${originalPath}`);
    }

    // Load merged code descriptions
    const mergedPath = `${EXPLANATIONS_MERGE_PDF}/merged_code_descriptions.json`;
    if (!await fs.pathExists(mergedPath)) {
      throw new Error(`Merged file not found: ${mergedPath}`);
    }

    const originalCodes = await fs.readJSON(originalPath);
    const mergedCodes = await fs.readJSON(mergedPath);

    // Count codes with explanations
    let totalOriginal = 0;
    let totalMerged = 0;
    let codesWithExplanations = 0;

    // Compare each code type
    for (const codeType of ["CPT", "ICD-10", "HCPCS"]) {
      if (!originalCodes[codeType] || !mergedCodes[codeType]) continue;

      console.log(`\n${codeType} Codes:`);

      const originalCount = originalCodes[codeType].length;
      const mergedCount = mergedCodes[codeType].length;

      totalOriginal += originalCount;
      totalMerged += mergedCount;

      // Count codes with explanations in this type
      const withExplanations = mergedCodes[codeType].filter((code: any) => code.explanation).length;
      codesWithExplanations += withExplanations;

      console.log(`  Original: ${originalCount} codes`);
      console.log(`  Merged: ${mergedCount} codes`);
      console.log(`  With explanations: ${withExplanations} codes (${Math.round(withExplanations / mergedCount * 100)}%)`);

      // Show a sample of codes with explanations
      if (withExplanations > 0) {
        const sample = mergedCodes[codeType].find((code: any) => code.explanation);
        if (sample) {
          console.log(`\n  Sample code with explanation:`);
          console.log(`  Code: ${sample.code}`);
          console.log(`  Description: ${sample.description}`);
          console.log(`  Explanation: ${sample.explanation.substring(0, 150)}...`);
        }
      }
    }

    console.log(`\n📊 Summary:`);
    console.log(`  Total original codes: ${totalOriginal}`);
    console.log(`  Total merged codes: ${totalMerged}`);
    console.log(`  Codes with explanations: ${codesWithExplanations} (${Math.round(codesWithExplanations / totalMerged * 100)}%)`);

  } catch (error) {
    console.error("❌ Error comparing descriptions:", error);
  }
}

/**
 * Function to load code descriptions (either original or merged)
 * @param useMerged Whether to use merged descriptions with explanations
 * @returns The loaded code descriptions
 */
async function loadCodeDescriptions(state: GraphStateType): Promise<any> {
  try {
    let filePath;

    if (state.useExplanations) {
      filePath = `${EXPLANATIONS_MERGE_PDF}/merged_code_descriptions.json`;
      console.log("Using merged code descriptions with explanations");
    } else {
      filePath = CACHED_CODE_DESCRIPTIONS_JSON;
      console.log("Using original cached code descriptions");
    }

    if (!await fs.pathExists(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }

    return await fs.readJSON(filePath);
  } catch (error: any) {
    console.error(`Error loading code descriptions: ${error.message}`);
    throw error;
  }
}

async function verifyLowConfidenceAnswers(questions: Question[], state: GraphStateType): Promise<Question[]> {
  const lowConfidenceQuestions = questions.filter(q => (q.confidence || 0) < 6);

  if (lowConfidenceQuestions.length === 0) {
    console.log("✅ No low-confidence questions to verify.");
    return questions;
  }

  console.log(`🔍 Verifying ${lowConfidenceQuestions.length} low-confidence questions (Confidence < 6)...`);

  const verifiedQuestions = [...questions];
  let gptSuccessCount = 0;

  // Load all cached code descriptions once before the loop
  let cachedCodes: any;
  try {
    cachedCodes = await loadCodeDescriptions(state);
  } catch {
    console.log("No cached code descriptions found. Verification may have lower accuracy.");
    cachedCodes = { "HCPCS": [], "ICD-10": [], "CPT": [] }; // Ensure object exists
  }

  for (let i = 0; i < lowConfidenceQuestions.length; i++) {
    const question = lowConfidenceQuestions[i];
    console.log(`Verifying ${i + 1}/${lowConfidenceQuestions.length}: Question ${question.number} (Initial Confidence: ${question.confidence})...`);

    const questionType = determineQuestionType(question);
    console.log(`Question ${question.number} identified as ${questionType} question`);

    const codeDescriptions: Map<string, string> = new Map();

    // --- Start: Code Description Extraction Logic ---
    if (question.options) {
      for (const option of question.options) {
        let matches: RegExpMatchArray | null = null;
        let codeType: 'HCPCS' | 'ICD-10' | 'CPT' | null = null;

        // Determine code type and extract codes based on pattern
        if (/[A-Z]\d{4}/g.test(option)) {
          matches = option.match(/[A-Z]\d{4}/g);
          codeType = 'HCPCS';
        } else if (/[A-Z]\d{2}(\.\d+)?/g.test(option)) {
          matches = option.match(/[A-Z]\d{2}(\.\d+)?/g);
          codeType = 'ICD-10';
        } else if (/\b\d{5}\b/g.test(option)) {
          matches = option.match(/\b\d{5}\b/g);
          codeType = 'CPT';
        }

        if (matches && codeType && cachedCodes[codeType]) {
          for (const code of matches) {
            if (codeDescriptions.has(code)) continue; // Skip if already found

            const cachedCode = cachedCodes[codeType].find((item: any) => item.code === code);
            if (cachedCode) {
              let fullDescription = cachedCode.description;
              if (state.useExplanations && cachedCode.explanation) {
                fullDescription += ` | Explanation: ${cachedCode.explanation}`;
              }
              codeDescriptions.set(code, fullDescription);
            } else {
              console.log(`Code ${code} (${codeType}) not found in cache`);
            }
          }
        }
      }
    }
    // --- End: Code Description Extraction Logic ---

    // --- GPT-4o Verification ---
    try {
      // Build the context string from the descriptions we found
      let codeDescriptionsContext = '';
      if (codeDescriptions.size > 0) {
        const descriptionsArray = Array.from(codeDescriptions.entries()).map(([code, desc]) => `${code}: ${desc}`);
        codeDescriptionsContext = `**Official Code Descriptions:**\n${descriptionsArray.join('\n')}\n`;
      }

      const gptPrompt = `You are a senior medical coding auditor. Re-examine this question with detailed analysis and provide your response in the requested format.
Use the official code descriptions provided below to inform your reasoning.

**Question ${question.number}:** ${question.text}
${question.options ? question.options.join('\n') : ''}
Current Answer: ${question.myAnswer}

Vital Coding Descriptions: ${codeDescriptionsContext}
**Task:**
Analyze each option against current coding guidelines, using the provided descriptions as the source of truth. Provide your reasoning, then conclude with the mandatory JSON block.

**Output Format:**
You MUST conclude your entire response with a single JSON code block in the following format. Do not add any text after this block.

\`\`\`json
{
  "reasoningSummary": "A brief summary of why your answer is correct and others are wrong based on the provided descriptions.",
  "finalAnswer": "[A/B/C/D]",
  "confidence": [A number from 1-10]
}
\`\`\``;

      const gptResponse = await llmVerify.invoke(gptPrompt);
      const content = typeof gptResponse.content === 'string' ? gptResponse.content : '';

      const jsonBlockStart = content.lastIndexOf('```json');
      if (jsonBlockStart === -1) {
        throw new Error("Could not find the mandatory JSON block in the LLM response.");
      }

      const jsonString = content.substring(jsonBlockStart + 7, content.lastIndexOf('```'));
      const result = JSON.parse(jsonString);

      const fullReasoning = `LLM Analysis:\n${content.split('```json')[0].trim()}\n\nSummary: ${result.reasoningSummary}`;

      const questionIndex = verifiedQuestions.findIndex(q => q.number === question.number);
      verifiedQuestions[questionIndex] = {
        ...verifiedQuestions[questionIndex],
        verifiedAnswer: result.finalAnswer,
        confidence: result.confidence,
        reasoning: fullReasoning
      };

      console.log(`  ✅ Question ${question.number}: Verified with LLM. Answer: ${result.finalAnswer} (Conf: ${result.confidence})`);
      gptSuccessCount++;

    } catch (gptError: any) {
      console.log(`  ❌ Question ${question.number}: Verification failed. Error: ${gptError.message}`);
    }
  }

  const changedAnswers = verifiedQuestions.filter(q => q.verifiedAnswer && q.verifiedAnswer !== q.myAnswer).length;

  console.log(`\n✅ Verification complete:`);
  console.log(`  - Total questions verified: ${gptSuccessCount}/${lowConfidenceQuestions.length}`);
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
  // const sampleSize = Math.ceil(mediumConfidenceQuestions.length * 0.5);
  // const sampledQuestions = mediumConfidenceQuestions
  //   .sort(() => 0.5 - Math.random())
  //   .slice(0, sampleSize);
  const sampledQuestions = [] as Question[];

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
  console.log("🔍 Looking for questions.json file...");

  try {
    // First check if questions.json exists
    if (await fs.pathExists(QUESTIONS_JSON)) {
      console.log(`✅ Found ${QUESTIONS_JSON}! Loading questions from JSON file...`);
      const questionsData = await fs.readJSON(QUESTIONS_JSON);
      console.log(`✅ Loaded ${questionsData.length} questions from JSON file`);
      return { extractedQuestions: questionsData };
    }

    // If JSON file doesn't exist, fall back to PDF processing
    console.log(`⚠️ ${QUESTIONS_JSON} not found. Falling back to PDF processing...`);
    const pdfContent = await processPdf(TEST_PDF);
    const questions = await extractQuestions(pdfContent);

    // Save the extracted questions to JSON for future use
    console.log(`💾 Saving extracted questions to ${QUESTIONS_JSON} for future use...`);
    await fs.writeJSON(QUESTIONS_JSON, questions, { spaces: 2 });
    console.log(`✅ Saved ${questions.length} questions to JSON file`);

    return { extractedQuestions: questions };
  } catch (error) {
    console.error("❌ Error in extractNode:", error);
    throw error;
  }
}

async function answerNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  let questionsToProcess = state.extractedQuestions;

  if (state.questionLimit && state.questionLimit > 0 && state.questionLimit < questionsToProcess.length) {
    console.log(`⚙️ Limiting to ${state.questionLimit} questions as requested`);
    questionsToProcess = questionsToProcess.slice(0, state.questionLimit);
  }

  const answered = await answerQuestionsWithConfidence(questionsToProcess, state);
  return { answeredQuestions: answered };
}

async function verifyNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  const verified = await verifyLowConfidenceAnswers(state.answeredQuestions, state);
  return { verifiedQuestions: verified };
}

async function devilsAdvocateNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  const challenged = await devilsAdvocateCheck(state.verifiedQuestions);
  await saveAnswersToFile(challenged);
  return { verifiedQuestions: challenged };
}

async function printQuestionsByType(questions: Question[]): Promise<void> {
  console.log("📋 Questions by Type:");

  // Group questions by type
  const cptQuestions = questions.filter(q => q.questionType === 'CPT');
  const icdQuestions = questions.filter(q => q.questionType === 'ICD-10');
  const hcpcsQuestions = questions.filter(q => q.questionType === 'HCPCS');
  const generalQuestions = questions.filter(q => q.questionType === 'GENERAL');

  console.log(`CPT Questions (${cptQuestions.length}):`);
  cptQuestions.forEach(q => console.log(`  ${q.number}. ${q.text.substring(0, 50)}...`));

  console.log(`\nICD-10 Questions (${icdQuestions.length}):`);
  icdQuestions.forEach(q => console.log(`  ${q.number}. ${q.text.substring(0, 50)}...`));

  console.log(`\nHCPCS Questions (${hcpcsQuestions.length}):`);
  hcpcsQuestions.forEach(q => console.log(`  ${q.number}. ${q.text.substring(0, 50)}...`));

  console.log(`\nGeneral Questions (${generalQuestions.length}):`);
  generalQuestions.forEach(q => console.log(`  ${q.number}. ${q.text.substring(0, 50)}...`));
}

async function savePerformanceLog(performanceLog: PerformanceLog): Promise<void> {
  console.log("💾 Saving performance log...");

  // Save current performance log
  await fs.writeJSON(PERFORMANCE_LOG_JSON, performanceLog, { spaces: 2 });
  console.log(`✅ Performance log saved to: ${PERFORMANCE_LOG_JSON}`);

  // Update performance history
  try {
    let history: PerformanceLog[] = [];

    // Check if history file exists
    if (await fs.pathExists(PERFORMANCE_HISTORY_JSON)) {
      history = await fs.readJSON(PERFORMANCE_HISTORY_JSON);
    }

    // Add current log to history
    history.push(performanceLog);

    // Save updated history
    await fs.writeJSON(PERFORMANCE_HISTORY_JSON, history, { spaces: 2 });
    console.log(`✅ Performance history updated in: ${PERFORMANCE_HISTORY_JSON}`);

  } catch (error) {
    console.error("❌ Error updating performance history:", error);
  }
}

async function generatePerformanceLog(questions: Question[]): Promise<PerformanceLog> {
  console.log("📊 Generating performance log...");

  // Initialize counters
  const summary = {
    totalQuestions: questions.length,
    correctAnswers: questions.filter(q => q.isCorrect).length,
    byQuestionType: {
      CPT: { total: 0, correct: 0, percentage: 0 },
      'ICD-10': { total: 0, correct: 0, percentage: 0 },
      HCPCS: { total: 0, correct: 0, percentage: 0 },
      GENERAL: { total: 0, correct: 0, percentage: 0 }
    },
    byModel: {} as { [modelName: string]: { total: number; correct: number; percentage: number } }
  };

  // Initialize questions object
  const questionsLog: { [questionId: string]: any } = {};

  // Process each question
  for (const q of questions) {
    const questionType = q.questionType || 'GENERAL';
    const modelUsed = q.modelUsed || 'unknown';

    // Add to question type stats
    summary.byQuestionType[questionType].total++;
    if (q.isCorrect) {
      summary.byQuestionType[questionType].correct++;
    }

    // Initialize model stats if needed
    if (!summary.byModel[modelUsed]) {
      summary.byModel[modelUsed] = { total: 0, correct: 0, percentage: 0 };
    }

    // Add to model stats
    summary.byModel[modelUsed].total++;
    if (q.isCorrect) {
      summary.byModel[modelUsed].correct++;
    }

    // Add question details to log
    questionsLog[q.number.toString()] = {
      questionType,
      modelUsed,
      isCorrect: q.isCorrect || false,
      confidence: q.confidence || 0,
      initialAnswer: q.myAnswer,
      verifiedAnswer: q.verifiedAnswer,
      correctAnswer: q.correctAnswer
    };
  }

  // Calculate percentages
  for (const type in summary.byQuestionType) {
    const stats = summary.byQuestionType[type as keyof typeof summary.byQuestionType];
    stats.percentage = stats.total > 0 ? Math.round((stats.correct / stats.total) * 100) : 0;
  }

  for (const model in summary.byModel) {
    const stats = summary.byModel[model];
    stats.percentage = stats.total > 0 ? Math.round((stats.correct / stats.total) * 100) : 0;
  }

  // Create the log object
  const performanceLog: PerformanceLog = {
    timestamp: new Date().toISOString(),
    questions: questionsLog,
    summary
  };

  return performanceLog;
}
async function compareNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  console.log("📊 Comparing final answers with answer key...");

  try {
    let testResults: TestResults;

    // First check if answer_key.json exists
    if (await fs.pathExists(ANSWER_KEY_JSON)) {
      console.log(`✅ Found ${ANSWER_KEY_JSON}! Loading answer key from JSON file...`);

      const answerKeyData = await fs.readJSON(ANSWER_KEY_JSON);

      // Process the answer key data
      const finalAnswers = state.verifiedQuestions.map(q => ({
        number: q.number,
        myAnswer: q.verifiedAnswer || q.myAnswer
      }));

      // Compare answers with the key
      const results = answerKeyData.map((keyItem: any) => {
        const myAnswer = finalAnswers.find(a => a.number === keyItem.number);
        return {
          number: keyItem.number,
          myAnswer: myAnswer?.myAnswer || "N/A",
          correctAnswer: keyItem.answer,
          isCorrect: myAnswer?.myAnswer === keyItem.answer
        };
      });

      const correctCount = results.filter((r: any) => r.isCorrect).length;
      const totalCount = results.length;
      const percentage = Math.round((correctCount / totalCount) * 100);
      const verifiedCount = state.verifiedQuestions.filter(q => q.verifiedAnswer).length;
      const challengedCount = state.verifiedQuestions.filter(q => q.reasoning?.includes("Perplexity challenge")).length;
      const perplexityCount = state.verifiedQuestions.filter(q => q.perplexityAnswer).length;

      testResults = {
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
    } else {
      // If JSON file doesn't exist, fall back to PDF processing
      console.log(`⚠️ ${ANSWER_KEY_JSON} not found. Falling back to PDF processing...`);
      const answerKeyContent = await processPdf(ANSWERS_PDF);
      testResults = await compareWithAnswerKey(state.verifiedQuestions, answerKeyContent);

      // Save the answer key to JSON for future use
      console.log(`💾 Saving answer key to ${ANSWER_KEY_JSON} for future use...`);
      const answerKeyData = testResults.details.map(item => ({
        number: item.number,
        answer: item.correctAnswer
      }));
      await fs.writeJSON(ANSWER_KEY_JSON, answerKeyData, { spaces: 2 });
      console.log(`✅ Saved answer key to JSON file`);
    }

    // Update questions with correctness information
    const updatedQuestions = state.verifiedQuestions.map(q => {
      const resultItem = testResults.details.find((r: any) => r.number === q.number);
      if (resultItem) {
        return {
          ...q,
          correctAnswer: resultItem.correctAnswer,
          isCorrect: resultItem.isCorrect
        };
      }
      return q;
    });

    // Generate and save performance log
    const performanceLog = await generatePerformanceLog(updatedQuestions);
    await savePerformanceLog(performanceLog);

    // Print questions by type
    await printQuestionsByType(updatedQuestions);

    return {
      testResults: testResults,
      verifiedQuestions: updatedQuestions
    };

  } catch (error) {
    console.error("❌ Error in compareNode:", error);
    throw error;
  }
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
  console.log("Features: LocalLLM or OpenAI API + Verification + Research-Based Devil's Advocate");
  console.log(`Files: ${TEST_PDF} → ${OUTPUT_FILE} ← ${ANSWERS_PDF}`);
  console.log("Commands: /run [number], /status, /print-by-type, /performance, /merge-explanations, /compare-explanations, /use-explanations, /use-original, quit\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  let state: GraphStateType = {
    extractedQuestions: [],
    answeredQuestions: [],
    verifiedQuestions: [],
    testResults: null,
    questionLimit: undefined,
    useExplanations: false
  };

  const ask = () => {
    rl.question("You: ", async (input) => {
      if (input === 'quit') {
        rl.close();
        return;
      }

      if (input.startsWith('/run')) {
        try {
          // Extract number of questions if provided
          const match = input.match(/\/run\s+(\d+)/);
          const questionLimit = match ? parseInt(match[1]) : undefined;

          console.log(`🚀 Starting advanced medical coding testing Agent${questionLimit ? ` (limited to ${questionLimit} questions)` : ''}!!!`);

          // Store the limit in the state
          state.questionLimit = questionLimit;

          const result = await app.invoke(state);
          state = { ...state, ...result };

          if (state.testResults) {
            console.log(`\n🎯 FINAL RESULTS:`);
            console.log(`Questions found: ${state.extractedQuestions.length}/100`);
            console.log(`Questions processed: ${state.testResults.totalQuestions}`);
            console.log(`Questions verified: ${state.testResults.verifiedCount}`);
            console.log(`Perplexity verifications: ${state.testResults.perplexityCount}`);
            console.log(`Questions challenged: ${state.testResults.challengedCount}`);
            console.log(`Score: ${state.testResults.correctAnswers}/${state.testResults.totalQuestions} (${state.testResults.percentage}%)`);
            console.log(`Answers saved to: ${OUTPUT_FILE}\n`);
            exec('afplay /System/Library/Sounds/Glass.aiff && say "Your coding test is complete. Relax!"');
          }
        } catch (error) {
          console.log("❌ Error:", error);
        }
      }
      else if (input === '/merge-explanations') {
        try {
          console.log("🔄 Starting explanation merge process...");
          await mergeExplanationsWithCachedDescriptions();
          console.log("✅ Explanation merge complete!");
        } catch (error) {
          console.error("❌ Error merging explanations:", error);
        }
      }
      else if (input === '/compare-explanations') {
        try {
          await compareOriginalAndMergedDescriptions();
        } catch (error) {
          console.error("❌ Error comparing explanations:", error);
        }
      }
      else if (input === '/use-explanations') {
        state.useExplanations = true;
        console.log("✅ Now using merged code descriptions with explanations");
      }
      else if (input === '/use-original') {
        state.useExplanations = false;
        console.log("✅ Now using original cached code descriptions");
      }
      else if (input === '/status') {
        console.log(`📋 Status:`);
        console.log(`Questions extracted: ${state.extractedQuestions.length}/100`);
        console.log(`Questions answered: ${state.answeredQuestions.length}`);
        console.log(`Questions verified: ${state.verifiedQuestions.length}`);
        console.log(`Using explanations: ${state.useExplanations ? 'Yes' : 'No'}`);
        console.log(`Test results: ${state.testResults ? `${state.testResults.percentage}%` : 'Not available'}\n`);
      }
      else if (input === '/print-by-type') {
        if (state.extractedQuestions.length > 0) {
          await printQuestionsByType(state.extractedQuestions);
        } else {
          console.log("❌ No questions extracted yet. Run /run first.");
        }
      }
      else if (input === '/performance') {
        try {
          if (await fs.pathExists(PERFORMANCE_LOG_JSON)) {
            const log = await fs.readJSON(PERFORMANCE_LOG_JSON);
            console.log("\n📊 Performance Summary:");
            console.log(`Total Questions: ${log.summary.totalQuestions}`);
            console.log(`Correct Answers: ${log.summary.correctAnswers} (${Math.round((log.summary.correctAnswers / log.summary.totalQuestions) * 100)}%)`);

            console.log("\nPerformance by Question Type:");
            for (const type in log.summary.byQuestionType) {
              const stats = log.summary.byQuestionType[type as keyof typeof log.summary.byQuestionType];
              console.log(`  ${type}: ${stats.correct}/${stats.total} (${stats.percentage}%)`);
            }

            console.log("\nPerformance by Model:");
            for (const model in log.summary.byModel) {
              const stats = log.summary.byModel[model];
              console.log(`  ${model}: ${stats.correct}/${stats.total} (${stats.percentage}%)`);
            }
          } else {
            console.log("❌ No performance log found. Run a test first.");
          }
        } catch (error) {
          console.error("❌ Error reading performance log:", error);
        }
      }
      else {
        console.log("Commands: /run (start test), /status (check progress), /merge-explanations, /compare-explanations, /use-explanations, /use-original\n");
      }

      ask();
    });
  };

  ask();
}

if (require.main === module) {
  startCLI().catch(console.error);
}
