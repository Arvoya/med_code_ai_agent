# README

This is v2 of an AI agent to achieve a high score on a practice medical code exam.

## Models Used

1. Gemma 2 9b
2. Gemma 3 1b
3. Gemma 3 12b
4. Gemma 3 27b
5. MedGemma 27b
6. gpt-4o
7. o3-mini
8. o3
9. o4-mini
10. Medical DeepSeek R1 fine-tuned
11. ii-medical-8b

I wanted to try models I could run locally, which is why I chose this variety.
Also researching into MedGemma, ii-medical-8b, and Medical DeepSeek showed promise
as the models might have more context around medical coding. Results later on said
a different story. I stopped using Perplexity as it was getting too pricey for
this project.

## Data Collection

### API's

1. [HCPCS](https://clinicaltables.nlm.nih.gov/apidoc/hcpcs/v3/doc.html)
2. [ICD-10](https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html)


### Web Crawled

1. [CPT Descriptions provided by AAPC](https://www.aapc.com/codes/cpt-codes-range/)

There is no free public available API for CPT codes, but the AMA does provide a [CPT
Developers Program](https://platform.ama-assn.org/ama/#/dev-program) that may be
worth looking into to enhance this agents accuracy.

### Explanations

The Answer PDF does provide explanations of the codes after each question, giving
more opportunity to give proper data relative to the time of the test *2023*.

## WorkFlow PipeLine

`Extract -> Answer -> Verify -> Compare`

Due to too many false positives I shortened the Pipeline in v2 vs v1 that had
additional `Challenge` node.

### Process

1. Parse Questions PDF
2. Save Questions & Multiple Choice to a JSON file
3. Answer Questions in Batches (Prevents overload tokens on various models)
   * Categorize Question `["HCPCS", "CPT", "ICD-10", "General"]`
   * Search for codes within Multiple Choice with Data JSON file
   * Retrieve any descriptions for found codes from Data JSON file
   * If none found, use API's/Web Crawl through AAPC website
   * Add Data to JSON file
   * Answer questions based on found Data with a Confidence Score `[1-10]`
   * Repeat for all questions
4. Re-Answer questions with `low (<6)` confidence score
   * Adjust prompt to really re-emphasize importance of found Data
   * Potential use of larger model if resources available
5. Parse Answers PDF
6. Save Answers to an Answer_Key JSON file
7. Compare Answers_Key with Agent Answers
8. Log Performance and Patterns of Model
9. Merge Explanation found within Answers PDF to New Explanation_Data JSON file
10. *Re-Run entire process with learned experience*

> \* Step **10** is optional in the model. I found the explanations within the Answers
PDF to result in more accurate answers. I believe its because the code data
relative to the time of the test. Potentially
some errors in test is due to Codes no longer existing or being changed.

## Performance Metrics

### Model Accuracy Comparison

| Model Name          | Overall Accuracy (%) | CPT (%) | HCPCS (%) | ICD-10 (%) | General (%) |
|---------------------|----------------------|---------|-----------|------------|-------------|
| Gemma 2 9b          | 68%                  | 62%     | 71%       | 69%        | 93%         |
| Gemma 3 1b          | 56%                  | 45%     | 100%      | 69%        | 73%         |
| Gemma 3 12b         | 86%                  | 82%     | 100%      | 92%        | 93%         |
| Gemma 3 27b         | 70%                  | 63%     | 71%       | 77%        | 93%         |
| MedGemma 27b        | 84%                  | 80%     | 86%       | 85%        | 100%        |
| gpt-4o              | 84%                  | 77%     | 86%       | 100%       | 100%        |
| **o3-mini**             | **90%**                  | **88%**     | **100%**      | **92%**        | **93%**         |
| o4-mini             | 86%                  | 82%     | 86%       | 92%        | 100%        |
| Medical DeepSeek R1 | 76%                  | 75%     | 86%       | 77%        | 73%         |
| ii-medical-8b       | 48%                  | 42%     | 14%       | 69%        | 80%         |

### Accuracy by Question Type

| Question Type | # of Questions | Accuracy Range | Best Models                   | Worst Models              |
|---------------|----------------|----------------|-------------------------------|---------------------------|
| CPT           | 65             | 42–88%         | o3-mini, o4-mini, Gemma 3 12b | ii-medical-8b, Gemma 3 1b |
| General       | 15             | 73-100%        | o3-mini, 04-mini, Gemma 3 12b | ii-medical-8b, Gemma 3 1b |
| ICD-10        | 13             | 69-100%        | o3-mini, gpt-4o, Gemma 3 12b  | ii-medical-8b, Gemma 3 1b |
| HCPCS         | 7              | 14-100%        | o3-mini, gpt-4o, Gemma 3 12b  | ii-medical-8b, Gemma 3 1b |


#### Insights

* **General and ICD-10:** These questions seem to be the easiest to answer
* **HCPCS:** Is a mixed bag, but mostly because ii-medical-8b with its outlier
score of 14% on these types of questions
* **CPT:** This is all around the most difficult to answer, my hunch is the yearly
changes to these and the difficulty for LLMs to access the data as its all behind
a paywall.

### Confidence Patterns

* **High Confidence = High Accuracy:** The best models (o3-mini, o4-mini, Gemma
3 12b) tend to have high confidence scores on correct answers, and lower confidence
on incorrect ones.
* **False Positives:** ii-medical-8b and Gemma 3 1b show high confidence on incorrect
answers, showing overconfidence and poor execution.

## Reflections

I think moving forward using o3-mini, o4-mini, Gemma 3 12b seem to be a good group
of models to play more with. [Hugging Face](https://huggingface.co/) is
a wonderful resource to explore and learn about other LLM's. I had high
hopes for ii-medical-8b but that and MedGemma could be better used for processing
medical images.

To get above 90% I would also look into the [CPT
Developers Program](https://platform.ama-assn.org/ama/#/dev-program) as CPT questions
make up 65% of entire exam.

## Quick Start

``` bash
npm install
npm start

#Commands:
/run [number] # Run all or a specified number of questions
/status # Deprecated command need to get rid of it
/print-by-type # Prints questions organized by category
/performance # After a run you can see more details
/merge-explanations # This will grab all explanations and add them to a new data file 
/compare-explanations # Will show what was changed in the merge
/use-explanations # This will make use of that new explanations data file 
/use-original # This will make ignore that new explanations data file and use the one without explanations
quit
```
