"""Bounded generation for the pinned Arena evaluator's existing endpoint.

Keep prompts, retries, answer cleanup and scoring identical to the upstream
evaluator. Save per-question evidence in the trial directory, not upstream data.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time


DEFAULT_MAX_WORKERS = 16


def generation_workers(config):
    value = int((config.get("custom") or {}).get("max_workers", 0))
    if value < 0:
        raise ValueError("custom.max_workers must be nonnegative")
    return value or DEFAULT_MAX_WORKERS


def generate_answers(module, args, *, workers, output_dir):
    questions = module.get_questions(args)
    server = module.VLLMServer(args, args.model_path)
    answers = {}
    output_path = output_dir / "answers.jsonl"
    diagnostics_path = output_dir / "generation.jsonl"

    def generate_one(question, endpoint):
        started = time.monotonic()
        payload = {
            "model": args.model_path,
            "messages": [{"role": "user", "content": question["prompt"]}],
            "max_tokens": args.max_new_tokens,
        }
        # Sessions are never shared between worker threads.
        with module.requests.Session() as session:
            api_key = os.environ.get("VLLM_API_KEY")
            if api_key:
                session.headers["Authorization"] = f"Bearer {api_key}"
            for attempt in range(1, module.VLLM_GENERATION_RETRY + 1):
                try:
                    response = session.post(
                        endpoint, json=payload, timeout=module.VLLM_REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    completion = response.json()
                    choices = completion.get("choices", [])
                    if not choices:
                        raise ValueError("vLLM response missing 'choices'.")
                    message = choices[0].get("message")
                    if not message:
                        raise ValueError("vLLM response missing message.")
                    content = (message.get("content") or "").strip()
                    reasoning = (message.get("reasoning_content") or "").strip()
                    if content:
                        answer_text = content
                    elif reasoning:
                        answer_text = reasoning
                    else:
                        answer_text = ""
                    break
                except (module.requests.RequestException, ValueError) as exc:
                    if attempt == module.VLLM_GENERATION_RETRY:
                        raise RuntimeError(
                            f"Failed to generate answer for uid {question['uid']} "
                            f"after {attempt} attempts"
                        ) from exc
                    time.sleep(2 ** attempt)

        if answer_text.startswith("<think>") and "</think>" in answer_text:
            answer_text = answer_text.split("</think>", maxsplit=1)[1].strip()
        elif "</think>" in answer_text:
            answer_text = answer_text.split("</think>", maxsplit=1)[1].strip()

        if not answer_text and reasoning:
            if "</think>" in reasoning:
                answer_text = reasoning.split("</think>", maxsplit=1)[1].strip()
            elif reasoning.startswith("<think>"):
                answer_text = reasoning[len("<think>"):].strip()
            else:
                answer_text = reasoning.strip()

        answer_text = module.limit_repetitions(answer_text)
        record = {
            "uid": question["uid"],
            "ans_id": module.shortuuid.uuid(),
            "model": args.model_alias,
            "messages": [
                {"role": "user", "content": question["prompt"]},
                {"role": "assistant", "content": {"answer": answer_text}},
            ],
            "tstamp": time.time(),
            "metadata": module._make_metadata(answer_text),
        }
        diagnostics = {
            "uid": question["uid"],
            "elapsed_seconds": time.monotonic() - started,
            "attempts": attempt,
            "finish_reason": choices[0].get("finish_reason"),
            "usage": completion.get("usage", {}),
        }
        return record, diagnostics

    try:
        port = server.start()
        endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
        workers = min(workers, max(len(questions), 1))
        print(f"[generate] questions={len(questions)} max_workers={workers}")
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as answers_file, \
                diagnostics_path.open("w", encoding="utf-8") as diagnostics_file, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(generate_one, q, endpoint): q for q in questions}
            try:
                for future in module.tqdm(
                    as_completed(futures), total=len(futures), desc="Generating answers"
                ):
                    record, diagnostics = future.result()
                    answers[record["uid"]] = record
                    answers_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    answers_file.flush()
                    diagnostics_file.write(json.dumps(diagnostics) + "\n")
                    diagnostics_file.flush()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        # Completion order must not change the upstream question ordering.
        return output_path, {q["uid"]: answers[q["uid"]] for q in questions}
    finally:
        server.stop()


def install_arena_runtime(module, config, output_dir):
    """Return actual completion counts and a callback restoring module hooks."""
    workers = generation_workers(config)
    original_generate = module.generate_answers
    original_judge = module.judge_answers
    counts = {}

    def generate(args):
        return generate_answers(module, args, workers=workers, output_dir=output_dir)

    def judge(args, candidate_answers=None):
        _, judgments = original_judge(args, candidate_answers)
        attempted = len(module.get_questions(args))
        completed = sum(
            1 for record in judgments
            if record is not None
            and len(record.get("games", [])) >= 2
            and all(game.get("score") is not None for game in record["games"][:2])
        )
        counts.update(attempted=attempted, completed=completed,
                      incomplete=attempted - completed)
        path = output_dir / "judgments.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in judgments:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path, judgments

    module.generate_answers = generate
    module.judge_answers = judge

    def restore():
        module.generate_answers = original_generate
        module.judge_answers = original_judge

    return counts, restore
