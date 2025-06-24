import axios from 'axios';
import * as dotenv from 'dotenv';

dotenv.config();

export class PerplexityAPI {
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

export const perplexity = new PerplexityAPI();
