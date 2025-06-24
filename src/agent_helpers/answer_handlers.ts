import { Question } from '../types'
import { GraphStateType } from '..';
import { loadCodeDescriptions } from '.';
import * as fs from 'fs-extra';
import * as cheerio from 'cheerio';
import axios from 'axios';
import { CACHED_CODE_DESCRIPTIONS_JSON } from '../config/constants';
import { extractKeywords, calculateMatchScore } from '../utils/functions';
import { llmAnswer } from '../config/models';


export async function handleHCPCSQuestion(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
  console.log(`Processing HCPCS question ${question.number}...`);

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

    const codeDescriptions: Map<string, string> = new Map();

    let cachedCodes: any;
    try {
      cachedCodes = await loadCodeDescriptions(state);
    } catch (error) {
      console.log("Creating new cache file with empty structure");
      cachedCodes = {
        "CPT": [],
        "ICD-10": [],
        "HCPCS": []
      };
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });
      console.log("New cache file created");
    }

    if (!cachedCodes["HCPCS"]) {
      console.log("HCPCS array missing in cache, initializing it");
      cachedCodes["HCPCS"] = [];
    }

    const codesToFetch: string[] = [];
    for (const code of hcpcsCodes) {
      const cachedCode = cachedCodes["HCPCS"].find((item: any) => item.code === code);
      if (cachedCode) {
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

    if (codesToFetch.length > 0) {
      console.log(`Fetching ${codesToFetch.length} HCPCS codes from API...`);

      for (const code of codesToFetch) {
        try {
          const response = await axios.get(`https://clinicaltables.nlm.nih.gov/api/hcpcs/v3/search?terms=${code}`);

          let description: string | null = null;

          if (response.data && Array.isArray(response.data)) {
            const data = response.data;

            if (data.length >= 4 && Array.isArray(data[3])) {

              const codeDescPairs = data[3];
              if (Array.isArray(codeDescPairs)) {
                for (const pair of codeDescPairs) {
                  if (Array.isArray(pair) && pair.length >= 2 && pair[0] === code) {
                    description = pair[1];
                    break;
                  }
                }
              }
            }
            else if (data[0] && Array.isArray(data[0])) {
              for (const pair of data) {
                if (Array.isArray(pair) && pair.length >= 2 && pair[0] === code) {
                  description = pair[1];
                  break;
                }
              }
            }
          }

          if (description) {
            codeDescriptions.set(code, description);

            cachedCodes["HCPCS"].push({
              code: code,
              description: description
            });

          } else {
            console.log(`No description found for code ${code} in API response`);
          }
        } catch (error) {
          console.error(`  Error querying HCPCS API for code ${code}:`, error);
        }
      }

      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });

      try {
        const verifyData = await fs.readJSON(CACHED_CODE_DESCRIPTIONS_JSON);
        console.log(`Cache file updated successfully. Contains ${verifyData["HCPCS"].length} HCPCS codes.`);
      } catch (error) {
        console.error("Error verifying cache file update:", error);
      }
    }

    let answer = '';
    let confidence = 6;
    let reasoning = '';

    const questionKeywords = extractKeywords(question.text);

    if (question.options) {
      for (let i = 0; i < question.options.length; i++) {
        const option = question.options[i];
        const optionLetter = String.fromCharCode(65 + i);

        const codeMatch = option.match(/[A-Z]\d{4}/);
        if (codeMatch) {
          const code = codeMatch[0];
          const description = codeDescriptions.get(code);

          if (description) {
            const matchScore = calculateMatchScore(description, questionKeywords);

            if (matchScore > 0.7) {
              answer = optionLetter;
              confidence = 8;
              reasoning = `HCPCS code ${code} description "${description}" matches the question context. Verification confirms this is the correct code.`;
              break;
            }
          }
        }
      }
    }

    if (!answer) {
      console.log(`USING ${llmAnswer.model} with code descriptions:`);





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

    console.log(`No HCPCS codes found in options for question ${question.number}, using ${llmAnswer.model}...`);
    await handleGeneralQuestion(question, answeredQuestions, state);
  }
}

export async function handleCPTQuestions(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
  console.log(`Processing CPT question ${question.number}...`);


  const cptCodes: string[] = [];
  if (question.options) {
    for (const option of question.options) {

      const matches = option.match(/\b\d{5}\b/g);
      if (matches) {
        cptCodes.push(...matches);
      }
    }
  }

  if (cptCodes.length > 0) {



    const codeDescriptions: Map<string, string> = new Map();


    let cachedCodes: any;
    try {
      cachedCodes = await loadCodeDescriptions(state);
    } catch (error) {

      console.log("Creating new cache file with empty structure");
      cachedCodes = {
        "CPT": [],
        "ICD-10": [],
        "HCPCS": []
      };
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });
      console.log("New cache file created");
    }


    if (!cachedCodes["CPT"]) {

      cachedCodes["CPT"] = [];
    }


    const codesToFetch: string[] = [];
    for (const code of cptCodes) {

      const cachedCode = cachedCodes["CPT"].find((item: any) => item.code === code);
      if (cachedCode) {
        if (state.useExplanations && cachedCode.explanation) {
          codeDescriptions.set(code, cachedCode.explanation);
        } else {
          codeDescriptions.set(code, cachedCode.description);
        }
      } else {

        codesToFetch.push(code);
      }
    }


    if (codesToFetch.length > 0) {
      console.log(`Fetching ${codesToFetch.length} CPT codes from AAPC website...`);

      for (const code of codesToFetch) {
        try {

          const url = `https://www.aapc.com/codes/cpt-codes/${code}`;
          const response = await axios.get(url);


          const $ = cheerio.load(response.data);


          const subHeadDetail = $('.sub_head_detail').text();


          let description: string | null = null;

          if (subHeadDetail) {


            const descMatch = subHeadDetail.match(/under the range - (.+?)\.$/);
            if (descMatch && descMatch[1]) {
              description = descMatch[1].trim();
            } else {

              description = subHeadDetail.trim();
            }
          }


          if (!description) {

            const codeTitle = $('h1.cpt_code').text().trim();
            if (codeTitle) {
              const titleMatch = codeTitle.match(/\d{5} (.+)$/);
              if (titleMatch && titleMatch[1]) {
                description = titleMatch[1].trim();
              }
            }
          }


          let summary = '';
          const cptLayterms = $('#cpt_layterms').find('p').first();
          if (cptLayterms.length > 0 && cptLayterms.text().trim()) {
            summary = cptLayterms.text().trim();
          } else {

            const offLongDesc = $('#offlongdesc').find('p').first();
            if (offLongDesc.length > 0 && offLongDesc.text().trim()) {
              summary = offLongDesc.text().trim();
            }
          }


          let fullDescription = description || '';
          if (summary && fullDescription) {
            fullDescription += ` Summary: ${summary}`;
          } else if (summary) {
            fullDescription = summary;
          }

          if (fullDescription) {
            codeDescriptions.set(code, fullDescription);



            cachedCodes["CPT"].push({
              code: code,
              description: fullDescription
            });


          } else {



            const placeholderDesc = `CPT code ${code} - research this code further for accurate information.`;
            codeDescriptions.set(code, placeholderDesc);


            cachedCodes["CPT"].push({
              code: code,
              description: placeholderDesc
            });


          }
        } catch (error: any) {
          console.error(`  Error fetching CPT code ${code} from website:`, error);


          let errorMessage = "";
          if (error.response && error.response.status === 404) {
            errorMessage = `CPT code ${code} - Not found on AAPC website. This may be a test code or requires specialized knowledge. Please research this code further.`;
          } else {
            errorMessage = `CPT code ${code} - Unable to retrieve description due to technical error. Please research this code further.`;
          }


          codeDescriptions.set(code, errorMessage);


          cachedCodes["CPT"].push({
            code: code,
            description: errorMessage
          });


        }


        console.log(`Waiting 3 seconds before next request...`);
        await new Promise(resolve => setTimeout(resolve, 3000));
      }


      console.log("Saving updated cache to file...");
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });


      try {
        await fs.readJSON(CACHED_CODE_DESCRIPTIONS_JSON);

      } catch (error) {
        console.error("Error verifying cache file update:", error);
      }
    }

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
    console.log(`No CPT codes found in options for question ${question.number}, using ${llmAnswer.model}...`);
    await handleGeneralQuestion(question, answeredQuestions, state);
  }
}

export async function handleICD10Question(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
  console.log(`Processing ICD-10 question ${question.number}...`);

  if (!question.options || !question.options.some(opt => /[A-Z]\d{2}(\.\d+)?/.test(opt))) {
    console.log(`No ICD-10 codes found in options for question ${question.number}, using ${llmAnswer.model}...`);
    await handleGeneralQuestion(question, answeredQuestions, state);
    return;
  }

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

    const codeDescriptions: Map<string, string> = new Map();

    let cachedCodes: any;
    try {
      cachedCodes = await loadCodeDescriptions(state);
    } catch (error) {

      console.log("Creating new cache file with empty structure");
      cachedCodes = {
        "CPT": [],
        "ICD-10": [],
        "HCPCS": []
      };
      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });
      console.log("New cache file created");
    }


    if (!cachedCodes["ICD-10"]) {

      cachedCodes["ICD-10"] = [];
    }


    const codesToFetch: string[] = [];
    for (const code of icdCodes) {

      const cachedCode = cachedCodes["ICD-10"].find((item: any) => item.code === code);
      if (cachedCode) {

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


    if (codesToFetch.length > 0) {

      for (const code of codesToFetch) {
        try {

          const response = await axios.get(`https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms=${code}`);


          let description: string | null = null;


          if (response.data && Array.isArray(response.data)) {
            const data = response.data;


            if (data.length >= 4 && Array.isArray(data[3])) {


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

            else if (data[0] && Array.isArray(data[0])) {

              for (const pair of data) {
                if (Array.isArray(pair) && pair.length >= 2 && pair[0] === code) {
                  description = pair[1];

                  break;
                }
              }
            }
          }

          if (description) {
            codeDescriptions.set(code, description);



            cachedCodes["ICD-10"].push({
              code: code,
              description: description
            });


          } else {
            console.log(`No description found for code ${code} in API response`);
          }
        } catch (error) {
          console.error(`  Error querying ICD-10 API for code ${code}:`, error);
        }
      }



      await fs.writeJSON(CACHED_CODE_DESCRIPTIONS_JSON, cachedCodes, { spaces: 2 });


      try {
        await fs.readJSON(CACHED_CODE_DESCRIPTIONS_JSON);

      } catch (error) {
        console.error("Error verifying cache file update:", error);
      }
    }


    const result = await useO4MiniForQuestion(question, codeDescriptions);

    answeredQuestions.push({
      ...question,
      myAnswer: result.answer,
      confidence: result.confidence,
      reasoning: result.reasoning,
      modelUsed: llmAnswer.model
    });

  } else {

    console.log(`No ICD-10 codes found in options for question ${question.number}, using ${llmAnswer.model}...`);
    await handleGeneralQuestion(question, answeredQuestions, state);
  }
}

export async function handleGeneralQuestion(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
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

    console.error(`Unexpected response format from ${llmAnswer.model}  for question ${question.number}`);
    answeredQuestions.push({
      ...question,
      myAnswer: "A",
      confidence: 5,
      reasoning: "Could not parse model response"
    });
  }
}


export async function useO4MiniForQuestion(
  question: Question,
  codeDescriptions: Map<string, string>
): Promise<{ answer: string, confidence: number, reasoning: string }> {

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

Vital Code Descriptions and Explanations from Official Database MAKE SURE YOU FOLLOW THIS:
${contextInfo}

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
    return {
      answer: answerMatch[1],
      confidence: parseInt(confidenceMatch[1]),
      reasoning: reasoningMatch ? reasoningMatch[1].trim() : ''
    };
  } else {

    console.error(`Unexpected response format from ${llmAnswer.model}`);
    return {
      answer: "A",
      confidence: 5,
      reasoning: "Could not parse model response"
    };
  }
}
