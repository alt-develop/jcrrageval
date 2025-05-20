import argparse
from gen_answer_local import gen_answer_local
from gen_answer_api import gen_answer_api
import os
import dotenv

dotenv.load_dotenv(verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate answers using a local model or API."
    )
    parser.add_argument(
        "--api",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Specify whether to use local model or API.",
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="wikirag",
        help="Name of the benchmark dataset.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for local model.",
    )
    parser.add_argument(
        "--base-url", type=str, default=None, help="Base URL for the API."
    )
    parser.add_argument(
        "--api_key", type=str, default=None, help="API key for model."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for sampling."
    )

    args = parser.parse_args()
    
    # check if model answer already exists
    model_answer_file = f"model_answer/{args.benchmark_name}/{args.model_path}.json"
    if os.path.exists(model_answer_file):
        print(f"Model answer file already exists for {args.model_path}. Skipping.")
        exit(0)

    if args.api == "local":
        gen_answer_local(
            args.model_path, 
            args.benchmark_name, 
            args.tensor_parallel_size
        )
    elif args.api == "api":
        gen_answer_api(
            args.model_path,
            args.benchmark_name,
            args.base_url,
            args.api_key,
            args.temperature,
        )
