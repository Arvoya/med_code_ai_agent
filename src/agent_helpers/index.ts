import * as fs from 'fs-extra';
import { Question, TestResults, PerformanceLog, EnrichArgs } from '../types';
import { GraphStateType } from '..'
import { llm, llmVerify } from '../config/models';
import { perplexity } from '../services/perplexity';
import { ANSWERS_PDF, EXPLANATIONS_MERGE_PDF, CACHED_CODE_DESCRIPTIONS_JSON, PERFORMANCE_LOG_JSON, PERFORMANCE_HISTORY_JSON } from '../config/constants';
import { handleCPTQuestions, handleHCPCSQuestion, handleICD10Question, handleGeneralQuestion } from './answer_handlers';
import { extractExplanationsFromAnswersPdf } from '../utils/pdf';



export async function extractQuestions(pdfContent: string): Promise<Question[]> {
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

export async function answerQuestionsWithConfidence(questions: Question[], state: GraphStateType): Promise<Question[]> {
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
        answeredQuestions.push({
          ...question,
          myAnswer: "A",
          confidence: 5,
          reasoning: "Error occurred during processing"
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

export function determineQuestionType(question: Question): 'HCPCS' | 'ICD-10' | 'CPT' | 'GENERAL' {
  const text = question.text.toLowerCase();
  const options = question.options ? question.options.join(' ').toLowerCase() : '';
  const fullText = text + ' ' + options;

  if (
    fullText.includes('hcpcs') ||
    fullText.includes('level ii code') ||
    fullText.includes('durable medical equipment') ||
    fullText.includes('prosthetic')
  ) {
    question.questionType = 'HCPCS';
    return 'HCPCS';
  }

  if (question.options) {
    for (const opt of question.options) {
      const match = opt.match(/[A-Z]\d{4}/);
      if (match && match[0] && match[0][0] !== 'C') {
        question.questionType = 'HCPCS';
        return 'HCPCS';
      }
    }
  }

  if (
    fullText.includes('icd-10') ||
    fullText.includes('diagnosis code') ||
    fullText.includes('diagnostic code') ||
    fullText.includes('according to icd')
  ) {
    question.questionType = 'ICD-10';
    return 'ICD-10';
  }

  if (question.options) {
    for (const opt of question.options) {
      if (opt.match(/[A-Z]\d{2}(\.\d+)?/)) {
        question.questionType = 'ICD-10';
        return 'ICD-10';
      }
    }
  }

  if (
    fullText.includes('cpt') ||
    fullText.includes('procedure code') ||
    fullText.includes('surgical code')
  ) {
    question.questionType = 'CPT';
    return 'CPT';
  }

  if (question.options) {
    for (const opt of question.options) {
      if (opt.match(/\b\d{5}\b/)) {
        question.questionType = 'CPT';
        return 'CPT';
      }
    }
  }

  question.questionType = 'GENERAL';
  return 'GENERAL';
}


export function extractCodesWithContext(text: string): Array<{ code: string, codeType: string, context: string }> {
  if (!text) return [];

  const results: Array<{ code: string, codeType: string, context: string }> = [];

  const patterns: { [key: string]: RegExp } = {
    "CPT": /\b(\d{5})\b/g,
    "ICD-10": /\b([A-Z]\d{2}(\.\d+)?)\b/g,
    "HCPCS": /\b([A-Z]\d{4})\b/g
  };

  for (const [codeType, pattern] of Object.entries(patterns)) {
    let match;
    const patternCopy = new RegExp(pattern);
    while ((match = patternCopy.exec(text)) !== null) {
      const code = match[1];

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

export async function mergeExplanationsWithCachedDescriptions(): Promise<any> {
  console.log("🔄 Merging explanations with cached code descriptions...");

  try {
    if (!await fs.pathExists(CACHED_CODE_DESCRIPTIONS_JSON)) {
      throw new Error(`Cached code descriptions file not found: ${CACHED_CODE_DESCRIPTIONS_JSON}`);
    }

    const cachedCodes = await fs.readJSON(CACHED_CODE_DESCRIPTIONS_JSON);
    console.log(`Loaded cached code descriptions with ${Object.values(cachedCodes).flat().length} entries`);

    const explanations = await extractExplanationsFromAnswersPdf();
    console.log(`Extracted ${explanations.length} explanations from answers PDF`);

    const mergedCodes = JSON.parse(JSON.stringify(cachedCodes));

    const explanationMap = new Map();
    explanations.forEach(exp => explanationMap.set(exp.number, exp));

    const codeExplanationMap = new Map();

    for (const explanation of explanations) {
      if (explanation.correctAnswer) {
        const codeMatch = explanation.correctAnswer.match(/[A-Z]\d{2}(\.\d+)?|\d{5}|[A-Z]\d{4}/);
        if (codeMatch) {
          const primaryCode = codeMatch[0];
          codeExplanationMap.set(primaryCode, explanation.explanation);
        }
      }

      const extractedCodes = extractCodesWithContext(explanation.explanation);

      for (const codeInfo of extractedCodes) {
        codeExplanationMap.set(codeInfo.code, explanation.explanation);
      }
    }

    let explanationsAdded = 0;
    let newCodesAdded = 0;

    for (const codeType of ["CPT", "ICD-10", "HCPCS"]) {
      if (!mergedCodes[codeType]) continue;

      for (const codeEntry of mergedCodes[codeType]) {
        if (codeExplanationMap.has(codeEntry.code)) {
          codeEntry.explanation = codeExplanationMap.get(codeEntry.code);
          explanationsAdded++;
        }
      }

      for (const [code, explanation] of codeExplanationMap.entries()) {
        let detectedCodeType: string;
        if (/^\d{5}$/.test(code)) {
          detectedCodeType = "CPT";
        } else if (/^[A-Z]\d{2}(\.\d+)?$/.test(code)) {
          detectedCodeType = "ICD-10";
        } else if (/^[A-Z]\d{4}$/.test(code)) {
          detectedCodeType = "HCPCS";
        } else {
          continue;
        }

        if (detectedCodeType !== codeType) continue;

        const codeExists = mergedCodes[codeType].some((entry: any) => entry.code === code);

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

    const dirPath = EXPLANATIONS_MERGE_PDF;
    await fs.ensureDir(dirPath);

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

export async function compareOriginalAndMergedDescriptions(): Promise<void> {
  console.log("🔍 Comparing original and merged code descriptions...");

  try {
    const originalPath = CACHED_CODE_DESCRIPTIONS_JSON;
    if (!await fs.pathExists(originalPath)) {
      throw new Error(`Original file not found: ${originalPath}`);
    }

    const mergedPath = `${EXPLANATIONS_MERGE_PDF}/merged_code_descriptions.json`;
    if (!await fs.pathExists(mergedPath)) {
      throw new Error(`Merged file not found: ${mergedPath}`);
    }

    const originalCodes = await fs.readJSON(originalPath);
    const mergedCodes = await fs.readJSON(mergedPath);

    let totalOriginal = 0;
    let totalMerged = 0;
    let codesWithExplanations = 0;

    for (const codeType of ["CPT", "ICD-10", "HCPCS"]) {
      if (!originalCodes[codeType] || !mergedCodes[codeType]) continue;

      console.log(`\n${codeType} Codes:`);

      const originalCount = originalCodes[codeType].length;
      const mergedCount = mergedCodes[codeType].length;

      totalOriginal += originalCount;
      totalMerged += mergedCount;

      const withExplanations = mergedCodes[codeType].filter((code: any) => code.explanation).length;
      codesWithExplanations += withExplanations;

      console.log(`  Original: ${originalCount} codes`);
      console.log(`  Merged: ${mergedCount} codes`);
      console.log(`  With explanations: ${withExplanations} codes (${Math.round(withExplanations / mergedCount * 100)}%)`);

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

export async function loadCodeDescriptions(state: GraphStateType): Promise<any> {
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


export async function verifyLowConfidenceAnswers(questions: Question[], state: GraphStateType): Promise<Question[]> {
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


export async function devilsAdvocateCheck(questions: Question[]): Promise<Question[]> {
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


export async function compareWithAnswerKey(myAnswers: Question[], answerKeyContent: string): Promise<TestResults> {
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

export async function savePerformanceLog(performanceLog: PerformanceLog): Promise<void> {
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

export async function enrichCodeDescriptionWithLLM({
  code,
  codeType,
  originalExplanation,
  questionText,
  options,
  aiReasoning
}: EnrichArgs): Promise<string> {
  const prompt = `
You are a medical coding expert. Here is a code description and a real exam scenario where an AI got the question wrong.

--- CODE DESCRIPTION ---
${code} - ${originalExplanation || "(none)"}

--- QUESTION SCENARIO ---
Question: ${questionText}
Options: ${options ? options.join('\n') : "(none)"}
AI's (incorrect) reasoning: ${aiReasoning || "(none)"}

--- TASK ---
Update and improve the code description to help future coders avoid this mistake.
-   Add clarifications, warnings, or use-case examples as needed.
-   If this code is often confused with another, explain how to tell them apart.
-   Make the explanation more robust and practical for real-world coding.

Return only the improved code description.
  `.trim();

  // Use your existing OpenAI or LangChain LLM setup
  const response = await llm.invoke(prompt);
  return typeof response.content === 'string' ? response.content.trim() : '';
}

export async function generatePerformanceLog(questions: Question[]): Promise<PerformanceLog> {
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
