# JCrRAG: Japanese Contextual relevance RAG Benchmark
A human-annotated benchmark for evaluating Japanese Retrieval-Augmented Generation (RAG) systems, featuring multi-level complexity and diverse categories.

## Benchmark Overview
### Complexity Levels
Based on the framework from [Zhao et al., 2024](https://arxiv.org/pdf/2409.14924):
![RAG Complexity Levels](rag-levels.png)

- **Level 1**: Direct fact retrieval
- **Level 2**: Multi-document synthesis
- **Level 3**: Complex reasoning and inference


### Categories
- History - Geography - Law
- Medical Care - Insurance - Finance
- Internal Regulations - Japanese Culture


## Evaluation Metrics

| Metric         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| Faithfulness   | Factual consistency with provided context                                  |
| Relevance      | Answer pertinence to the question                                          |
| Completeness   | Coverage of all required information elements                              |
| Utilization    | Effective use of contextual evidence                                       |

## Getting Started

### Prerequisites
- Python 3.10+
- pip package manager

### Installation
```bash
git clone https://github.com/ngocnhq/JCrRAG.git
cd JCrRAG
pip install -r requirements.txt
```

### Dataset Preparation
Download the dataset from...
Prepare the input data file in CSV/JSON format.

Run the data preparation script:
```bash
python prepare_data.py \
  --input_file path/to/your_data.csv \
  --context_col "context" \ # Context column name
  --question_col "question" \ # Question column name
  --answer_col "answer" \ # Ground truth column name
  --max_samples 1000 \ # Maximum samples to process
  --task "rag"
```
Processed data will be saved to `./data/`


## Model Evaluation
### Local vLLM Models

1. Generate answers:
```bash
   python gen_answer.py \
   --model_path $MODEL_PATH \
   --api local \
   --tensor_parallel_size $NUM_GPUS \
   --benchmark_name $BENCHMARK_NAME
```

### API-based Models
1. Generate answers (OpenAI-compatible API):

```bash
   python gen_answer.py \
  --benchmark_name $BENCHMARK_NAME \
  --model_path $MODEL_NAME \
  --api api \
  --base-url $API_ENDPOINT \
  --api-key $API_KEY
```

## Evaluation Judgement
Generate assessment scores using GPT-4o as judge:
```bash
   python gen_judgement.py \
   --model_list $YOUR_MODEL_NAME \
   --judge-models gpt-4o \
   --benchmark_name $BENCHMARK_NAME
```