import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { ANSWERS_PDF, OUTPUT_FILE } from "../config/constants";
import { llm } from "../config/models";
import * as fs from 'fs-extra';
import { Question } from "../types";

export async function processPdf(filePath: string): Promise<string> {
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

export async function extractExplanationsFromAnswersPdf(): Promise<any[]> {
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

export async function saveAnswersToFile(questions: Question[]): Promise<void> {
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

