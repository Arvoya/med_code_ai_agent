export interface Question {
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

export interface TestResults {
  totalQuestions: number;
  correctAnswers: number;
  incorrectAnswers: number;
  percentage: number;
  details: Question[];
  verifiedCount: number;
  challengedCount: number;
  perplexityCount: number;
}

export interface EnrichArgs {
  code: string;
  codeType: string;
  originalExplanation: string;
  questionText: string;
  options: string[];
  aiReasoning: string;
}

export interface PerformanceLog {
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


