#!/usr/bin/env python
"""
Batch Sentiment Analysis Script

Supports multiple methods:
- LLM: Local LLM via Ollama (qwen3.5:4b)
- SVM: Traditional ML baseline (SVM + TF-IDF)

Usage:
    # Single text with LLM (default)
    python batch_sentiment.py --text "Great product!"

    # Single text with SVM
    python batch_sentiment.py --text "Great product!" --method svm

    # Batch processing
    python batch_sentiment.py --input reviews.jsonl --output results.jsonl

    # Compare methods
    python batch_sentiment.py --input reviews.jsonl --output results.jsonl --compare

    # Health check
    python batch_sentiment.py --health-check
"""

import sys
import json
import csv
import time
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from local_llm.inference.predictor import OllamaPredictor
from local_llm.inference.schemas import LLMPrediction, BatchPredictionResult
from baseline.sentiment.svm_classifier import SVMSentimentClassifier, SVMConfig


# ============== Language Detection ==============

def detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return 'zh'
    if any('\u0400' <= c <= '\u04ff' for c in text):
        return 'ru'
    return 'en'


# ============== Data Loading ==============

def load_jsonl(file_path: str, text_field: str = 'text') -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records.append(record)
    return records


def load_csv(file_path: str, text_column: str) -> List[Dict[str, Any]]:
    """Load data from CSV file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_column in row:
                records.append({
                    'text': row[text_column],
                    'original': row
                })
    return records


def save_jsonl(results: List[Dict[str, Any]], file_path: str):
    """Save results to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


# ============== Model Loaders ==============

def get_model_dir() -> Path:
    """Get models directory."""
    return Path(__file__).parent.parent.parent / 'models'


def load_svm_model(language: str = 'en') -> Optional[SVMSentimentClassifier]:
    """Load pre-trained SVM model for language."""
    model_path = get_model_dir() / f'svm_{language}.pkl'
    if model_path.exists():
        classifier = SVMSentimentClassifier()
        classifier.load(str(model_path))
        return classifier
    return None


def get_llm_predictor(model_name: str = 'qwen3.5:4b') -> OllamaPredictor:
    """Get LLM predictor."""
    return OllamaPredictor(model_name=model_name, use_few_shot=True, n_few_shot_examples=3)


# ============== Unified Prediction Interface ==============

class UnifiedPredictor:
    """Unified interface for LLM and SVM methods."""

    def __init__(self, method: str = 'llm', model_name: str = 'qwen3.5:4b'):
        self.method = method
        self.model_name = model_name
        self.llm_predictor = None
        self.svm_models = {}

        if method == 'llm' or method == 'compare':
            self.llm_predictor = get_llm_predictor(model_name)

        if method == 'svm' or method == 'compare':
            # Lazy load SVM models
            pass

    def get_svm_model(self, language: str) -> Optional[SVMSentimentClassifier]:
        """Get or load SVM model for language."""
        if language not in self.svm_models:
            self.svm_models[language] = load_svm_model(language)
        return self.svm_models[language]

    def predict_llm(self, text: str, language: str) -> Dict[str, Any]:
        """Predict using LLM."""
        start_time = time.time()
        prediction = self.llm_predictor.predict(text, language)
        latency_ms = (time.time() - start_time) * 1000

        return {
            'method': 'llm',
            'sentiment_label': int(prediction.sentiment_label),
            'sentiment': ['negative', 'neutral', 'positive'][prediction.sentiment_label],
            'confidence': round(prediction.confidence, 4),
            'rationale': prediction.rationale,
            'latency_ms': round(latency_ms, 2),
            'model': prediction.model_name
        }

    def predict_svm(self, text: str, language: str) -> Dict[str, Any]:
        """Predict using SVM."""
        start_time = time.time()

        model = self.get_svm_model(language)
        if model is None:
            return {
                'method': 'svm',
                'error': f'No SVM model found for language: {language}. Run: python cloud_agent/scripts/train_svm_baseline.py',
                'sentiment_label': None
            }

        # SVM predict
        import numpy as np
        label = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        latency_ms = (time.time() - start_time) * 1000

        return {
            'method': 'svm',
            'sentiment_label': int(label),
            'sentiment': ['negative', 'neutral', 'positive'][label],
            'confidence': round(float(proba[label]), 4),
            'probabilities': {
                'negative': round(float(proba[0]), 4),
                'neutral': round(float(proba[1]), 4),
                'positive': round(float(proba[2]), 4)
            },
            'latency_ms': round(latency_ms, 2),
            'model': f'svm_{language}'
        }

    def predict(self, text: str, language: str = 'auto') -> Dict[str, Any]:
        """Predict using configured method."""
        if language == 'auto':
            language = detect_language(text)

        if self.method == 'llm':
            return self.predict_llm(text, language)
        elif self.method == 'svm':
            return self.predict_svm(text, language)
        elif self.method == 'compare':
            # Run both methods
            llm_result = self.predict_llm(text, language)
            svm_result = self.predict_svm(text, language)

            return {
                'text': text,
                'language': language,
                'llm': llm_result,
                'svm': svm_result,
                'agreement': llm_result['sentiment_label'] == svm_result.get('sentiment_label')
            }

        raise ValueError(f"Unknown method: {self.method}")

    def health_check(self) -> Dict[str, Any]:
        """Check health of configured methods."""
        results = {}

        if self.method in ('llm', 'compare'):
            llm_healthy = self.llm_predictor.health_check()
            results['llm'] = {
                'status': 'healthy' if llm_healthy else 'unhealthy',
                'model': self.model_name,
                'ready': llm_healthy
            }

        if self.method in ('svm', 'compare'):
            svm_status = {}
            for lang in ['en', 'zh', 'ru']:
                model = self.get_svm_model(lang)
                svm_status[lang] = model is not None and model.is_fitted
            results['svm'] = {
                'status': 'healthy' if any(svm_status.values()) else 'unhealthy',
                'models': svm_status,
                'ready': any(svm_status.values())
            }

        return results


# ============== Analysis Functions ==============

def analyze_single(
    predictor: UnifiedPredictor,
    text: str,
    language: str = 'auto'
) -> Dict[str, Any]:
    """Analyze single text."""
    return predictor.predict(text, language)


def analyze_batch(
    predictor: UnifiedPredictor,
    texts: List[str],
    language: str = 'auto',
    batch_size: int = 8,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """Analyze batch of texts."""
    results = []
    total = len(texts)
    start_time = time.time()

    for i, text in enumerate(texts):
        try:
            result = analyze_single(predictor, text, language)
            result['text'] = text
            results.append(result)

            if show_progress and (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%) | "
                      f"Rate: {rate:.2f} texts/s | ETA: {eta:.0f}s", file=sys.stderr)
        except Exception as e:
            results.append({
                'text': text,
                'error': str(e),
                'sentiment_label': None
            })

    return results


def generate_report(results: List[Dict[str, Any]], method: str = 'llm') -> Dict[str, Any]:
    """Generate statistics report."""
    # Handle compare mode separately
    if method == 'compare':
        return generate_compare_report(results)

    valid = [r for r in results if 'error' not in r and r.get('sentiment_label') is not None]

    if not valid:
        return {
            'method': method,
            'total': len(results),
            'valid': 0,
            'errors': len(results)
        }

    # Handle compare mode
    if method == 'compare':
        return generate_compare_report(results)

    # Sentiment distribution
    sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
    for r in valid:
        sentiment_counts[r['sentiment']] += 1

    # Language distribution
    lang_counts = {}
    for r in valid:
        lang = r.get('language', 'unknown')
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    # Confidence statistics
    confidences = [r['confidence'] for r in valid]
    avg_confidence = sum(confidences) / len(confidences)

    # Latency statistics
    latencies = [r['latency_ms'] for r in valid]
    avg_latency = sum(latencies) / len(latencies)
    total_latency = sum(latencies)

    return {
        'method': method,
        'total': len(results),
        'valid': len(valid),
        'errors': len(results) - len(valid),
        'sentiment_distribution': sentiment_counts,
        'sentiment_percentages': {
            k: round(v/len(valid)*100, 1) for k, v in sentiment_counts.items()
        },
        'language_distribution': lang_counts,
        'avg_confidence': round(avg_confidence, 4),
        'avg_latency_ms': round(avg_latency, 2),
        'total_latency_s': round(total_latency / 1000, 2),
        'processing_rate': round(len(valid) / (total_latency / 1000), 2) if total_latency > 0 else 0
    }


def generate_compare_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comparison report for LLM vs SVM."""
    # Valid results have both llm and svm keys and no top-level error
    valid = [r for r in results if 'llm' in r and 'svm' in r and 'error' not in r]

    if not valid:
        # Check what went wrong
        llm_errors = sum(1 for r in results if 'llm' not in r or 'error' in r.get('llm', {}))
        svm_errors = sum(1 for r in results if 'svm' not in r or 'error' in r.get('svm', {}))
        return {
            'method': 'compare',
            'total': len(results),
            'valid': 0,
            'errors': len(results),
            'llm_errors': llm_errors,
            'svm_errors': svm_errors
        }

    # Agreement statistics
    agreements = [r['agreement'] for r in valid]
    agreement_rate = sum(agreements) / len(agreements)

    # Per-method statistics
    llm_sentiments = [r['llm']['sentiment'] for r in valid]
    svm_sentiments = [r['svm'].get('sentiment', 'unknown') for r in valid]

    llm_dist = {'negative': 0, 'neutral': 0, 'positive': 0}
    svm_dist = {'negative': 0, 'neutral': 0, 'positive': 0}

    for s in llm_sentiments:
        if s in llm_dist:
            llm_dist[s] += 1
    for s in svm_sentiments:
        if s in svm_dist:
            svm_dist[s] += 1

    # Latency comparison
    llm_latencies = [r['llm']['latency_ms'] for r in valid]
    svm_latencies = [r['svm'].get('latency_ms', 0) for r in valid]

    # Confidence comparison
    llm_confidences = [r['llm']['confidence'] for r in valid]
    svm_confidences = [r['svm'].get('confidence', 0) for r in valid]

    return {
        'method': 'compare',
        'total': len(results),
        'valid': len(valid),
        'errors': len(results) - len(valid),
        'agreement_rate': round(agreement_rate * 100, 1),
        'agreement_count': sum(agreements),
        'disagreement_count': len(agreements) - sum(agreements),
        'llm': {
            'sentiment_distribution': llm_dist,
            'avg_confidence': round(sum(llm_confidences) / len(llm_confidences), 4),
            'avg_latency_ms': round(sum(llm_latencies) / len(llm_latencies), 2),
            'total_latency_s': round(sum(llm_latencies) / 1000, 2)
        },
        'svm': {
            'sentiment_distribution': svm_dist,
            'avg_confidence': round(sum(svm_confidences) / len(svm_confidences), 4),
            'avg_latency_ms': round(sum(svm_latencies) / len(svm_latencies), 2),
            'total_latency_s': round(sum(svm_latencies) / 1000, 2)
        },
        'speedup': round(sum(llm_latencies) / max(sum(svm_latencies), 1), 2)
    }


def print_report(report: Dict[str, Any]):
    """Print report to stderr."""
    print("\n" + "="*60, file=sys.stderr)
    print("SENTIMENT ANALYSIS REPORT", file=sys.stderr)
    print("="*60, file=sys.stderr)

    print(f"\nTotal: {report['total']}", file=sys.stderr)
    print(f"Valid: {report['valid']}", file=sys.stderr)
    print(f"Errors: {report['errors']}", file=sys.stderr)

    if report['method'] == 'compare':
        print(f"\n{'─'*60}", file=sys.stderr)
        print("COMPARISON: LLM vs SVM", file=sys.stderr)
        print(f"{'─'*60}", file=sys.stderr)

        print(f"\nAgreement Rate: {report['agreement_rate']}%", file=sys.stderr)
        print(f"  Agreed: {report['agreement_count']}", file=sys.stderr)
        print(f"  Disagreed: {report['disagreement_count']}", file=sys.stderr)

        print(f"\n{'Method':<12} {'Negative':<12} {'Neutral':<12} {'Positive':<12}", file=sys.stderr)
        print(f"{'─'*48}", file=sys.stderr)

        llm = report['llm']['sentiment_distribution']
        svm = report['svm']['sentiment_distribution']
        print(f"{'LLM':<12} {llm['negative']:<12} {llm['neutral']:<12} {llm['positive']:<12}", file=sys.stderr)
        print(f"{'SVM':<12} {svm['negative']:<12} {svm['neutral']:<12} {svm['positive']:<12}", file=sys.stderr)

        print(f"\n{'Metric':<20} {'LLM':<15} {'SVM':<15}", file=sys.stderr)
        print(f"{'─'*50}", file=sys.stderr)
        print(f"{'Avg Confidence':<20} {report['llm']['avg_confidence']:<15} {report['svm']['avg_confidence']:<15}", file=sys.stderr)
        print(f"{'Avg Latency (ms)':<20} {report['llm']['avg_latency_ms']:<15} {report['svm']['avg_latency_ms']:<15}", file=sys.stderr)
        print(f"{'Total Time (s)':<20} {report['llm']['total_latency_s']:<15} {report['svm']['total_latency_s']:<15}", file=sys.stderr)
        print(f"\nSVM Speedup: {report['speedup']}x faster", file=sys.stderr)
    else:
        print(f"\nMethod: {report['method'].upper()}", file=sys.stderr)
        print(f"\nSentiment Distribution:", file=sys.stderr)
        for s, p in report['sentiment_percentages'].items():
            print(f"  {s}: {p}%", file=sys.stderr)

        print(f"\nAvg Confidence: {report['avg_confidence']}", file=sys.stderr)
        print(f"Avg Latency: {report['avg_latency_ms']}ms", file=sys.stderr)
        print(f"Total Time: {report['total_latency_s']}s", file=sys.stderr)
        print(f"Processing Rate: {report['processing_rate']} texts/s", file=sys.stderr)

    print("\n" + "="*60, file=sys.stderr)


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description='Batch Sentiment Analysis')
    parser.add_argument('--text', type=str, help='Single text to analyze')
    parser.add_argument('--input', type=str, help='Input file (JSONL or CSV)')
    parser.add_argument('--output', type=str, help='Output file (JSONL)')
    parser.add_argument('--text-column', type=str, default='text',
                        help='Text column name for CSV files')
    parser.add_argument('--language', type=str, default='auto',
                        choices=['auto', 'zh', 'en', 'ru'],
                        help='Language code or auto-detect')
    parser.add_argument('--method', type=str, default='llm',
                        choices=['llm', 'svm', 'compare'],
                        help='Analysis method: llm (Ollama), svm (baseline), compare (both)')
    parser.add_argument('--model', type=str, default='qwen3.5:4b',
                        help='LLM model name')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for progress display')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress display')
    parser.add_argument('--health-check', action='store_true',
                        help='Check if models are ready')
    parser.add_argument('--report', action='store_true',
                        help='Generate statistics report')
    parser.add_argument('--save-raw', type=str,
                        help='Save raw predictions for later analysis')

    args = parser.parse_args()

    # Initialize predictor
    predictor = UnifiedPredictor(method=args.method, model_name=args.model)

    # Health check
    if args.health_check:
        health = predictor.health_check()
        print(json.dumps(health, indent=2))

        all_ready = all(
            h.get('ready', False) for h in health.values()
        )
        sys.exit(0 if all_ready else 1)

    # Single text mode
    if args.text:
        result = analyze_single(predictor, args.text, args.language)
        result['text'] = args.text
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)

    # Batch mode from file
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        # Load data
        if args.input.endswith('.jsonl'):
            records = load_jsonl(args.input)
            texts = [r.get('text', r.get('original_text', '')) for r in records]
        elif args.input.endswith('.csv'):
            records = load_csv(args.input, args.text_column)
            texts = [r['text'] for r in records]
        else:
            print(f"Error: Unsupported file format: {args.input}", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(texts)} texts from {args.input}", file=sys.stderr)
        print(f"Method: {args.method.upper()}", file=sys.stderr)

        # Process
        results = analyze_batch(
            predictor, texts, args.language,
            args.batch_size, not args.no_progress
        )

        # Generate report
        if args.report:
            report = generate_report(results, args.method)
            print_report(report)

        # Save raw predictions
        if args.save_raw:
            save_jsonl(results, args.save_raw)
            print(f"\nRaw predictions saved to {args.save_raw}", file=sys.stderr)

        # Save or print results
        if args.output:
            save_jsonl(results, args.output)
            print(f"\nResults saved to {args.output}", file=sys.stderr)
        else:
            print("\n=== Results ===")
            for result in results:
                print(json.dumps(result, ensure_ascii=False))

        sys.exit(0)

    # No input
    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()