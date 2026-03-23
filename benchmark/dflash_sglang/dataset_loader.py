"""Dataset loader for DFlash benchmarks.

Extracted from DFlash/model/utils.py — only the load_and_process_dataset function.
"""
from __future__ import annotations

import os

from datasets import Features, Sequence, Value, load_dataset


def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x['input'] else x['instruction'])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]
        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"
        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features
        )

    elif data_name == "longbench":
        allowed_files = [
            "narrativeqa.jsonl", "qasper.jsonl", "multifieldqa_en.jsonl", "multifieldqa_zh.jsonl",
            "hotpotqa.jsonl", "2wikimqa.jsonl", "musique.jsonl", "dureader.jsonl",
            "gov_report.jsonl", "qmsum.jsonl", "multi_news.jsonl", "vcsum.jsonl",
            "trec.jsonl", "triviaqa.jsonl", "samsum.jsonl", "lsht.jsonl",
            "passage_count.jsonl", "passage_retrieval_en.jsonl", "passage_retrieval_zh.jsonl",
            "lcc.jsonl", "repobench-p.jsonl",
        ]
        data_dir = "LongBench/data"
        local_files = [os.path.join(data_dir, fn) for fn in allowed_files]
        existing_files = [f for f in local_files if os.path.exists(f)]
        longbench_features = Features({
            "input": Value("string"),
            "context": Value("string"),
            "answers": Sequence(Value("string")),
            "length": Value("int32"),
            "dataset": Value("string"),
            "language": Value("string"),
            "all_classes": Sequence(Value("string")),
            "_id": Value("string"),
        })
        dataset = load_dataset(
            "json",
            data_files={"test": existing_files},
            split="test",
            features=longbench_features
        )

        SUBSET_PROMPTS = {
            "narrativeqa": (
                "You are given a story, which can be either a novel or a movie script, "
                "and a question. Answer the question as concisely as you can, using a single "
                "phrase if possible. Do not provide any explanation.\n\n"
                "Story: {context}\n\nNow, answer the following question based on the above story:\n\n"
                "Question: {input}\n\nAnswer:"
            ),
            "qasper": (
                "You are given a scientific article and a question. Answer the question as "
                "concisely as you can, using a single phrase or sentence if possible. If the "
                "question cannot be answered based on the information in the article, write "
                "\"unanswerable\". If the question is a yes/no question, answer \"yes\", "
                "\"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
                "Article: {context}\n\nAnswer the following question based on the above article:\n\n"
                "Question: {input}\n\nAnswer:"
            ),
            "multifieldqa_en": (
                "Read the following text and answer briefly.\n\n{context}\n\n"
                "Now, answer the following question based on the above text, only give me "
                "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
            ),
            "multifieldqa_zh": (
                "阅读以下文字并用中文简短回答：\n\n{context}\n\n"
                "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n"
                "问题：{input}\n答案："
            ),
            "hotpotqa": (
                "Answer the question based on the given passages. Only give me the answer "
                "and do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
                "Answer the question based on the given passages. Only give me the answer "
                "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
            ),
            "2wikimqa": (
                "Answer the question based on the given passages. Only give me the answer "
                "and do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
                "Answer the question based on the given passages. Only give me the answer "
                "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
            ),
            "musique": (
                "Answer the question based on the given passages. Only give me the answer "
                "and do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
                "Answer the question based on the given passages. Only give me the answer "
                "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
            ),
            "dureader": (
                "请基于以下给定文章回答问题，只告诉我答案，不要输出任何其他字词。\n\n"
                "文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n答案："
            ),
            "gov_report": (
                "You are given a report by a government agency. Write a one-page summary "
                "of the report.\n\nReport:\n{context}\n\n"
                "Now, write a one-page summary of the report.\n\nSummary:"
            ),
            "qmsum": (
                "You are given a meeting transcript and a query containing the topic of a "
                "meeting summary. Write a summary of the meeting transcript based on the query.\n\n"
                "Transcript:\n{context}\n\nQuery: {input}\n\nSummary:"
            ),
            "multi_news": (
                "You are given several news passages. Write a one-page summary of all news.\n\n"
                "Passages:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
            ),
            "vcsum": "下面有一段会议记录，请用1段话总结会议的内容。\n\n会议记录：\n{context}\n\n总结：",
            "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
            "triviaqa": (
                "Answer the question based on the given passage. Only give me the answer "
                "and do not output any other words. The following are some examples.\n\n{context}\n\n{input}"
            ),
            "samsum": "Summarize the given dialogue. Here are some examples.\n\n{context}\n\n{input}",
            "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
            "passage_count": (
                "There are some paragraphs below sourced from Wikipedia. Some of them may be "
                "duplicates. Please carefully read these paragraphs and determine how many unique "
                "paragraphs there are after removing duplicates. In other words, how many "
                "non-repeating paragraphs are there in total?\n\n{context}\n\n"
                "Please enter the final count of unique paragraphs after removing duplicates. "
                "The answer should be a positive integer.\n\nAnswer:"
            ),
            "passage_retrieval_en": (
                "Here are 30 paragraphs from Wikipedia, along with an abstract of a paragraph. "
                "Please determine which paragraph the abstract is from.\n\n{context}\n\n"
                "The following is an abstract.\n\n{input}\n\n"
                "Please enter the number of the paragraph that the abstract is from. The answer "
                "format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nAnswer:"
            ),
            "passage_retrieval_zh": (
                "以下是若干段落文字，以及一个段落的摘要。请确定摘要出自哪个段落。\n\n{context}\n\n"
                "以下是一个摘要\n\n{input}\n\n"
                "请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案："
            ),
            "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
            "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
        }

        def format_longbench(doc):
            template = SUBSET_PROMPTS[doc["dataset"]]
            return template.format(context=doc["context"], input=doc.get("input", ""))

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_longbench(x)]},
            remove_columns=dataset.column_names,
            features=target_features
        )

    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    return dataset
