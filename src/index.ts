import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import * as readline from 'readline';
import * as fs from 'fs-extra';
import * as dotenv from 'dotenv';
import { exec } from 'child_process';
import { Question, TestResults } from './types';
import { OUTPUT_FILE, ANSWERS_PDF, TEST_PDF, PERFORMANCE_LOG_JSON, ANSWER_KEY_JSON, EXPLANATIONS_MERGE_PDF, QUESTIONS_JSON } from './config/constants';
import { compareOriginalAndMergedDescriptions, compareWithAnswerKey, verifyLowConfidenceAnswers, enrichCodeDescriptionWithLLM, savePerformanceLog, mergeExplanationsWithCachedDescriptions, devilsAdvocateCheck, extractQuestions, answerQuestionsWithConfidence, generatePerformanceLog } from './agent_helpers'
import { processPdf, saveAnswersToFile } from './utils/pdf'

dotenv.config();


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
    default: () => true
  })
});

export type GraphStateType = typeof GraphState.State;



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

async function enrichNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
  // 1. Early exit if all answers are correct
  if (!state.testResults || state.testResults.incorrectAnswers === 0) {
    console.log("🎉 No incorrect answers! No enrichment needed.");
    return {};
  }

  // 2. Load merged code descriptions
  const mergedPath = `${EXPLANATIONS_MERGE_PDF}/merged_code_descriptions.json`;
  if (!await fs.pathExists(mergedPath)) {
    console.error("❌ Merged code descriptions not found!");
    return {};
  }
  const codeDescriptions = await fs.readJSON(mergedPath);

  // 3. Build set of incorrect question numbers
  const incorrectNumbers = new Set(
    state.testResults.details
      .filter(q => !q.isCorrect)
      .map(q => q.number)
  );

  // 4. Enrich only incorrect questions
  for (const q of state.verifiedQuestions) {
    if (!incorrectNumbers.has(q.number) || !q.options) continue;

    // Find correct answer letter for this question
    const result = state.testResults.details.find(d => d.number === q.number);
    const correctLetter = result?.correctAnswer;
    if (!correctLetter) continue;

    // Find the correct option and extract the code
    const correctOption = q.options.find(opt => opt.trim().startsWith(correctLetter + '.'));
    const codeMatch = correctOption?.match(/[A-Z]\.\s*([\w\.]+)/);
    const correctCode = codeMatch ? codeMatch[1] : null;
    if (!correctCode) continue;

    // Determine code type and set
    let codeSet: any[] | null = null, codeType = '';
    if (/^\d{5}$/.test(correctCode)) { codeSet = codeDescriptions.CPT; codeType = 'CPT'; }
    else if (/^[A-Z]\d{2}/.test(correctCode)) { codeSet = codeDescriptions['ICD-10']; codeType = 'ICD-10'; }
    else if (/^[A-Z]\d{4}$/.test(correctCode)) { codeSet = codeDescriptions.HCPCS; codeType = 'HCPCS'; }
    if (!codeSet) continue;

    const codeEntry = codeSet.find((c: any) => c.code === correctCode);
    if (!codeEntry) continue;

    // Call LLM to enrich the explanation
    const improvedExplanation = await enrichCodeDescriptionWithLLM({
      code: correctCode,
      codeType,
      originalExplanation: codeEntry.explanation || codeEntry.description,
      questionText: q.text,
      options: q.options,
      aiReasoning: q.reasoning || ''
    });

    if (improvedExplanation) {
      codeEntry.explanation = improvedExplanation;
      console.log(`🧠 Updated explanation for ${codeType} code ${correctCode}`);
    }
  }

  // 5. Save updated code descriptions
  await fs.writeJSON(mergedPath, codeDescriptions, { spaces: 2 });
  console.log(`✅ Codebook enriched with new scenario-based explanations.`);

  return {};
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
  .addNode("enrich", enrichNode)
  .addEdge(START, "extract")
  .addEdge("extract", "answer")
  .addEdge("answer", "verify")
  .addEdge("verify", "challenge")
  .addEdge("challenge", "compare")
  .addEdge("compare", "enrich")
  .addEdge("enrich", END);

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
