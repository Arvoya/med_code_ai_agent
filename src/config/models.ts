import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from 'dotenv';

dotenv.config();

export function useLocalModels(): boolean {
  return process.env.USE_LOCAL_MODELS === 'true';
}

export const llm = useLocalModels()
  ? new ChatOpenAI({
    model: process.env.LOCAL_MODEL,
    temperature: 0.1,
    openAIApiKey: "lm-studio",
    configuration: {
      baseURL: process.env.LM_STUDIO_URL || "http://localhost:1234/v1",
    },
  })
  : new ChatOpenAI({
    model: "o3",
  });

export const llmVerify = useLocalModels()
  ? new ChatOpenAI({
    model: process.env.LOCAL_MODEL,
    openAIApiKey: "lm-studio",
    configuration: {
      baseURL: process.env.LM_STUDIO_URL || "http://localhost:1234/v1",
    },
  })
  : new ChatOpenAI({
    model: "o3",
  });

export const llmAnswer = useLocalModels()
  ? new ChatOpenAI({
    model: process.env.LOCAL_MODEL,
    openAIApiKey: "lm-studio",
    configuration: {
      baseURL: process.env.LM_STUDIO_URL || "http://localhost:1234/v1",
    },
  })
  : new ChatOpenAI({
    model: "o3",
  });
