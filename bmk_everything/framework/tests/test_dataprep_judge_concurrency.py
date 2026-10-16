import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from core import dataprep_eval as runner
from core import provider


@pytest.fixture
def endpoints(monkeypatch):
    values = {
        'MODEL_NAME': 'candidate', 'OPENAI_API_BASE': 'http://candidate/v1',
        'OPENAI_API_TOKEN': 'candidate-token', 'JUDGE_MODEL': 'judge',
        'JUDGE_OPENAI_API_BASE': 'http://judge/v1', 'JUDGE_OPENAI_API_TOKEN': 'judge-token',
        'KIMI_PROVIDER_RETRIES': '0',
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    for key in ['KIMI_FINAL_ONLY', 'KIMI_THINKING_TYPE', 'JUDGE_MIN_TOKENS']:
        monkeypatch.delenv(key, raising=False)
    return {key: os.environ.get(key) for key in values}


def completion(content, reason='stop'):
    return {'choices': [{'message': {'content': content}, 'finish_reason': reason}], 'usage': {}}


def items(n=6):
    return [{'index': i, 'complete': True, 'finish_reason': 'stop',
             'response': f'<think>reason-{i}</think><answer>d{i}</answer>',
             'record': {'messages': [{'role': 'user', 'content': str(i)}],
                        'metadata': {'final_diagnosis': f'd{i}',
                                     'diagnostic_reasoning': '1. point-a 2. point-b'}}}
            for i in range(n)]


def test_parallel_judging_is_bounded_and_matches_serial(monkeypatch, endpoints):
    barrier = threading.Barrier(3, timeout=5)
    lock = threading.Lock()
    active = peak = 0
    parallel = False

    def request(endpoint, path, payload, timeout):
        nonlocal active, peak
        assert endpoint.base_url == 'http://judge/v1'
        assert endpoint.key() == 'judge-token'
        assert payload['model'] == 'judge'
        assert {key: os.environ.get(key) for key in endpoints} == endpoints
        with lock:
            active += 1
            peak = max(peak, active)
        if parallel:
            barrier.wait()
        with lock:
            active -= 1
        prompt = payload['messages'][0]['content']
        if 'Clinician point:' in prompt:
            answer = 'y' if 'point-a' in prompt else 'n'
        else:
            index = int(re.search(r'Predicted diagnosis: d(\d+)', prompt)[1])
            answer = 'y' if index % 2 == 0 else 'n'
        return completion(answer)

    monkeypatch.setattr(provider, '_request', request)
    serial_rows = items()
    serial = runner._score_generated('medcasereasoning', serial_rows, 'candidate', {}, 1)
    assert peak == 1
    parallel = True
    concurrent_rows = items()
    concurrent = runner._score_generated('medcasereasoning', concurrent_rows, 'candidate', {}, 3)
    assert peak == 3
    assert concurrent == serial
    assert concurrent_rows == serial_rows
    score, meta = runner.aggregate('medcasereasoning', concurrent[0])
    assert score == 0.5
    assert meta['reasoning_recall'] == 0.5
    assert {key: os.environ.get(key) for key in endpoints} == endpoints


def test_candidate_and_judge_can_overlap_without_cross_routing(monkeypatch, endpoints):
    barrier = threading.Barrier(2, timeout=5)
    observed = []

    def request(endpoint, path, payload, timeout):
        barrier.wait()
        observed.append((endpoint.base_url, endpoint.key(), payload['model']))
        return completion('y')

    monkeypatch.setattr(provider, '_request', request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(runner._judge, [{'role': 'user', 'content': 'judge'}], 'candidate', {})
        b = pool.submit(provider.chat_detailed, [{'role': 'user', 'content': 'candidate'}])
        assert a.result() == 'y'
        assert b.result().content == 'y'
    assert sorted(observed) == [('http://candidate/v1', 'candidate-token', 'candidate'),
                                ('http://judge/v1', 'judge-token', 'judge')]
    assert {key: os.environ.get(key) for key in endpoints} == endpoints


def test_judge_retry_keeps_explicit_endpoint_and_model(monkeypatch, endpoints):
    seen = []
    monkeypatch.setenv('KIMI_PROVIDER_RETRIES', '1')
    monkeypatch.setattr(provider.time, 'sleep', lambda _: None)

    def request(endpoint, path, payload, timeout):
        seen.append((endpoint.base_url, endpoint.key(), payload['model']))
        if len(seen) == 1:
            raise provider.ProviderError('rate limit', status=429)
        return completion('y')

    monkeypatch.setattr(provider, '_request', request)
    assert runner._judge([], 'candidate', {}) == 'y'
    assert seen == [('http://judge/v1', 'judge-token', 'judge')] * 2


def test_parallel_failure_and_truncation_keep_reward_contract(tmp_path, monkeypatch, endpoints):
    records = [(r['index'], r['record']) for r in items(3)]
    monkeypatch.setattr(runner, '_load', lambda *a: records)

    def request(endpoint, path, payload, timeout):
        if endpoint.base_url == 'http://candidate/v1':
            index = int(payload['messages'][0]['content'])
            return completion(f'<answer>d{index}</answer>', 'length' if index == 2 else 'stop')
        if 'Predicted diagnosis: d1' in payload['messages'][0]['content']:
            raise provider.ProviderError('judge failed', status=401)
        return completion('y')

    monkeypatch.setattr(provider, '_request', request)
    config = tmp_path / 'config.yaml'
    output = tmp_path / 'output'
    config.write_text(yaml.safe_dump({'general': {'model': 'candidate', 'output_dir': str(output)},
                                     'custom': {'max_workers': 2, 'judge_workers': 2}}))
    root = Path(__file__).resolve().parents[2]
    runner.run(root / 'format/medcasereasoning', config, 3)
    reward = json.loads((output / 'reward.json').read_text())
    assert reward['reward'] is None
    meta = reward['meta_data']
    assert (meta['attempted'], meta['completed'], meta['scored'], meta['incomplete'], meta['truncated_as_incorrect']) == (3, 1, 2, 1, 1)
    assert meta['observed_reward'] == 0.5
    assert meta['judge_workers'] == 2
    assert meta['generation_elapsed_s'] >= 0 and meta['scoring_elapsed_s'] >= 0
    samples = [json.loads(l) for l in (output / 'samples.jsonl').read_text().splitlines()]
    assert [r['index'] for r in samples] == [0, 1, 2]
    assert samples[0]['score']['correct']
    assert 'judge failed' in samples[1]['judge_error']
    assert samples[2]['scored_as_incorrect']
    assert {key: os.environ.get(key) for key in endpoints} == endpoints


@pytest.mark.parametrize('workers', [0, -1])
def test_invalid_judge_workers_fail_before_generation(tmp_path, monkeypatch, workers):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'custom': {'judge_workers': workers}}))
    def forbidden(*args, **kwargs):
        raise AssertionError('must fail before loading data or sending requests')
    monkeypatch.setattr(runner, '_load', forbidden)
    with pytest.raises(ValueError, match='judge_workers must be positive'):
        runner.run(Path(__file__).resolve().parents[2] / 'format/medcasereasoning', config, 3)


@pytest.mark.parametrize('configured,expected', [(None, 16), (1, 1), (4, 4)])
def test_default_and_explicit_judge_workers_reach_scoring(tmp_path, monkeypatch, endpoints, configured, expected):
    records = [(r['index'], r['record']) for r in items(20)]
    monkeypatch.setattr(runner, '_load', lambda *a: records)
    monkeypatch.setattr(provider, '_request', lambda *a: completion('answer'))
    observed = []
    def score(benchmark, generated, model, custom, workers):
        observed.append(workers)
        for item in generated:
            item['score'] = {'correct': True, 'reasoning_recall': 1.0}
        return [item['score'] for item in generated], 0, 0
    monkeypatch.setattr(runner, '_score_generated', score)
    custom = {} if configured is None else {'judge_workers': configured}
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'general': {'output_dir': str(tmp_path / 'output')}, 'custom': custom}))
    runner.run(Path(__file__).resolve().parents[2] / 'format/medcasereasoning', config, 20)
    assert observed == [expected]


def test_explicit_judge_endpoint_never_uses_anthropic_or_fallback(monkeypatch, endpoints):
    monkeypatch.setenv('JUDGE_MODEL', 'claude-test')
    monkeypatch.setenv('ANTHROPIC_API_BASE', 'http://anthropic/v1')
    monkeypatch.setenv('ANTHROPIC_API_TOKEN', 'unused')
    seen = []
    def request(endpoint, path, payload, timeout):
        seen.append((endpoint.base_url, payload['model']))
        raise provider.ProviderError('content_filter')
    monkeypatch.setattr(provider, '_request', request)
    def forbidden(*args, **kwargs):
        raise AssertionError('explicit judge endpoint must not use Anthropic routing')
    monkeypatch.setattr(provider, '_anthropic_request', forbidden)
    with pytest.raises(provider.ProviderError, match='content_filter'):
        runner._judge([], 'candidate', {})
    assert seen == [('http://judge/v1', 'claude-test')]
