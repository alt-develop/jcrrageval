import json

def prompt_1_a(document, question, label, answer):
    return f"""Assign a score to evaluate the STUDENT ANSWER in the following Japanese reading comprehension problem. The problem consists of a CONTEXT and a QUESTION. The CONTEXT contains one or more text paragraphs, which provides clues to answer the QUESTION. The CONTEXT and QUESTION might be in the domain of social science (e.g., Japanese language, literature, history, geography) or natural science (e.g., physics, chemistry, mathematics). A Japanese student created an answer to the question. We call this answer the STUDENT ANSWER to distinguish from the correct answer, which is called the GROUND TRUTH. The GROUND TRUTH is the accurate and complete answer to the QUESTION based on the information provided in the CONTEXT. The assigned score must evaluate the STUDENT ANSWER in 4 aspects : relevance, faithfulness, completeness and utilization.
## Rules for evaluating the STUDENT ANSWER:
1. Relevance: Does the STUDENT ANSWER address the QUESTION?
- **Partially Relevance**: The STUDENT ANSWER addresses some parts of the QUESTION but misses or only partially addresses others.
- **Fully Relevant**: The STUDENT ANSWER directly and comprehensively addresses the entire QUESTION.
2. Faithfulness : Is the STUDENT ANSWER completely based on the CONTEXT, entirely accurate and consistent with the provided CONTEXT?. It validates that the STUDENT ANSWER is consistent with the CONTEXT and isn't using information other than what exists in the CONTEXT.
3. Completeness: Does the STUDENT ANSWER provide all the information needed to answer the QUESTION?
- **Partially Complete**: The STUDENT ANSWER covers some aspects of the QUESTION but omits important details.
- **Fully Complete**: The STUDENT ANSWER covers all aspects of the QUESTION in detail.
4. Utilization : Is the STUDENT ANSWER made up of information from all relevant chunks in the CONTEXT, if it is required to aggregate information from multiple relevant chunks to get a correct and complete answer?. The goal is to determine the extent to which each chunk is part of the STUDENT ANSWER.
## Steps to evaluate the STUDENT ANSWER:
1. **Understand the user's intent**: Explain in your own words what the intent of the QUESTION is.
2. **Check if the STUDENT ANSWER is correct**: Think step-by-step whether the STUDENT ANSWER correctly answers the QUESTION by comparing with the GROUND TRUTH.
3. **Evaluate the quality of the ANSWER**: Evaluate the quality of the ANSWER based on its relevance, faithfulness, completeness and utilization.
Assign a score: each with a single score between 0 and 2, where 2 is the highest score on that aspect:
- "relevance"
0: The ANSWER is not relevant to the QUESTION.
1: The ANSWER is partially relevant to the QUESTION.
2: The ANSWER is fully relevant to the QUESTION.
- "faithfulness"
0: The STUDENT ANSWER is not based on the information provided in the CONTEXT.
1: Some information in the STUDENT ANSWER is correct and is based on the information provided in the CONTEXT.
2: All information in the STUDENT ANSWER is correct and is based on the information provided in the CONTEXT.
- "completeness"
0: The STUDENT ANSWER does not provide enough information to answer the QUESTION.
1: The STUDENT ANSWER only answers some aspects of the QUESTION.
2: The STUDENT ANSWER fully answers the QUESTION.
- "utilization"
0: The STUDENT ANSWER does not use any chunk or sentence in the CONTEXT.
1: The STUDENT ANSWER only uses some chunks or sentences from the CONTEXT, compared with the GROUND TRUTH.
2: The STUDENT ANSWER uses exactly identical chunks or sentences in the CONTEXT with the GROUND TRUTH.
The last line of your evaluation must be the following format: 
1. relevance: a single score between 0 and 2.
2. faithfulness: a single score between 0 and 2.
3. completeness: a single score between 0 and 2.
4. utilization: a single score between 0 and 2.

[DOCUMENT] 
{document}
[QUESTION] 
{question}
[GROUND TRUTH] 
{label}
[STUDENT ANSWER]
{answer}"""

def prompt_1_a_r1(document, question, label, answer):
    return f"""Assign a score to evaluate the STUDENT ANSWER in the following Japanese reading comprehension problem. The problem consists of a CONTEXT and a QUESTION. The CONTEXT contains one or more text paragraphs, which provides clues to answer the QUESTION. The CONTEXT and QUESTION might be in the domain of social science (e.g., Japanese language, literature, history, geography) or natural science (e.g., physics, chemistry, mathematics). A Japanese student created an answer to the question. We call this answer the STUDENT ANSWER to distinguish from the correct answer, which is called the GROUND TRUTH. The GROUND TRUTH is the accurate and complete answer to the QUESTION based on the information provided in the CONTEXT. The assigned score must evaluate the STUDENT ANSWER in 4 aspects: relevance, faithfulness, completeness and utilization.

## Rules for evaluating the STUDENT ANSWER:
1. Relevance: Does the STUDENT ANSWER address the QUESTION?
- **Partially Relevance**: The STUDENT ANSWER addresses some parts of the QUESTION but misses or only partially addresses others.
- **Fully Relevant**: The STUDENT ANSWER directly and comprehensively addresses the entire QUESTION.
2. Faithfulness : Is the STUDENT ANSWER completely based on the CONTEXT, entirely accurate and consistent with the provided CONTEXT?. It validates that the STUDENT ANSWER is consistent with the CONTEXT and isn't using information other than what exists in the CONTEXT.
3. Completeness: Does the STUDENT ANSWER provide all the information needed to answer the QUESTION?
- **Partially Complete**: The STUDENT ANSWER covers some aspects of the QUESTION but omits important details.
- **Fully Complete**: The STUDENT ANSWER covers all aspects of the QUESTION in detail.
4. Utilization : Is the STUDENT ANSWER made up of information from all relevant chunks in the CONTEXT, if it is required to aggregate information from multiple relevant chunks to get a correct and complete answer?. The goal is to determine the extent to which each chunk is part of the STUDENT ANSWER.
## Steps to evaluate the STUDENT ANSWER:
1. **Understand the user's intent**: Explain in your own words what the intent of the QUESTION is.
2. **Check if the STUDENT ANSWER is correct**: Think step-by-step whether the STUDENT ANSWER correctly answers the QUESTION by comparing with the GROUND TRUTH.
3. **Evaluate the quality of the ANSWER**: Evaluate the quality of the ANSWER based on its relevance, faithfulness, completeness and utilization.
Assign a score: each with a single score between 0 and 2, where 2 is the highest score on that aspect:
- "relevance"
0: The ANSWER is not relevant to the QUESTION.
1: The ANSWER is partially relevant to the QUESTION.
2: The ANSWER is fully relevant to the QUESTION.
- "faithfulness"
0: The STUDENT ANSWER is not based on the information provided in the CONTEXT.
1: Some information in the STUDENT ANSWER is correct and is based on the information provided in the CONTEXT.
2: All information in the STUDENT ANSWER is correct and is based on the information provided in the CONTEXT.
- "completeness"
0: The STUDENT ANSWER does not provide enough information to answer the QUESTION.
1: The STUDENT ANSWER only answers some aspects of the QUESTION.
2: The STUDENT ANSWER fully answers the QUESTION.
- "utilization"
0: The STUDENT ANSWER does not use any chunk or sentence in the CONTEXT.
1: The STUDENT ANSWER only uses some chunks or sentences from the CONTEXT, compared with the GROUND TRUTH.
2: The STUDENT ANSWER uses exactly identical chunks or sentences in the CONTEXT with the GROUND TRUTH.
The last line of your evaluation must be the following format: 
1. relevance: a single score between 0 and 2.
2. faithfulness: a single score between 0 and 2.
3. completeness: a single score between 0 and 2.
4. utilization: a single score between 0 and 2.

[DOCUMENT] 
{document}
[QUESTION] 
{question}
[GROUND TRUTH] 
{label}
[STUDENT ANSWER]
{answer}

The evaluation of the STUDENT ANSWER must be in the json format and be placed in json block, for example:
```json
{{
  "relevance": 1,
  "faithfulness": 1,
  "completeness": 1,
  "utilization": 1
}}
```
"""


def prompt_2_a(document, question, answer):
    return f"""You are an impartial judge tasked with evaluating the quality of the EXPLANATION provided by a Japanese student for a reading comprehension problem in social science (e.g., Japanese language, literature, history, geography) or natural science (e.g., physics, chemistry, math). You will be given a QUESTION, one or more DOCUMENTS, and an EXPLANATION. The EXPLANATION is generated by the Japanese student based on the QUESTION and the DOCUMENTS. The DOCUMENTS do not contain the answer to the QUESTION. The goal of the EXPLANATION is to clarify why the DOCUMENTS do not contain the answer to the QUESTION.

Your task is to evaluate the EXPLANATION's quality based on relevance, clarity, and logical reasoning.

## Rules for evaluating an EXPLANATION:
1. Relevance: Does the EXPLANATION address the QUESTION and the DOCUMENTS?
   - **0**: The EXPLANATION is irrelevant to the QUESTION or DOCUMENTS.
   - **1**: The EXPLANATION partially addresses the QUESTION or DOCUMENTS.
   - **2**: The EXPLANATION is fully relevant to the QUESTION and DOCUMENTS.

2. Clarity: Is the EXPLANATION clear and understandable?
   - **0**: The EXPLANATION is unclear or confusing.
   - **1**: The EXPLANATION is somewhat clear but may contain ambiguities.
   - **2**: The EXPLANATION is very clear and easy to understand.

3. Logical Reasoning: Does the EXPLANATION logically and accurately explain why the DOCUMENTS do not contain the answer?
   - **0**: The EXPLANATION lacks logical reasoning or is incorrect.
   - **1**: The EXPLANATION shows some logical reasoning but may have flaws.
   - **2**: The EXPLANATION is logically sound and accurately reasons why the answer is not in the DOCUMENTS.

## Steps to evaluate an EXPLANATION:
1. **Understand the user's intent**: Explain in your own words what the user's intent is, given the QUESTION.
2. **Check if the EXPLANATION is correct**: Think step-by-step whether the EXPLANATION correctly explains why the DOCUMENTS do not contain the answer.
3. **Evaluate the quality of the EXPLANATION**: Evaluate the quality of the EXPLANATION based on its relevance, clarity, and logical reasoning. 

Assign a score: Produce a single-line JSON object with the following keys, each with a single score between 0 and 2, where 2 is the highest score on that aspect:
- "relevance"
- "clarity"
- "logical_reasoning"

No additional instructions or explanations are allowed.

QUESTION: 
{question}

DOCUMENTS:
{document}

EXPLANATION:
{answer}
"""
