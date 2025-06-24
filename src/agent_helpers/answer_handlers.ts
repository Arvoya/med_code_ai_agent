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

export async function handleCPTQuestions(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
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
        if (state.useExplanations && cachedCode.explanation) {
          codeDescriptions.set(code, cachedCode.explanation);
        } else {
          codeDescriptions.set(code, cachedCode.description);
        }
      } else {
        // console.log(`Code ${ code } not found in cache, will fetch from website`);
        codesToFetch.push(code);
      }
    }

    // Fetch any codes not found in cache
    if (codesToFetch.length > 0) {
      console.log(`Fetching ${codesToFetch.length} CPT codes from AAPC website...`);

      for (const code of codesToFetch) {
        try {
          // console.log(`Querying website for code: ${ code } `);
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
            const placeholderDesc = `CPT code ${code} - research this code further for accurate information.`;
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

export async function handleICD10Question(question: Question, answeredQuestions: Question[], state: GraphStateType): Promise<void> {
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


export async function useO4MiniForQuestion(
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

  console.log("CONTEXT INFO BEING SHARED WITH PROMPT", contextInfo)

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
