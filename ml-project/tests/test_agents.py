"""
tests/test_agents.py — Unit tests for the Multi-Agent Q&A synthetic data pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.generator.agents.critic_agent import CriticAgent
from src.generator.agents.refiner_agent import RefinerAgent
from src.generator.agents.generator_agent import GeneratorAgent
from src.generator.agents.orchestrator import MultiAgentOrchestrator


class TestCriticAgent:
    def test_evaluate_success(self, config):
        agent = CriticAgent(config)
        mock_response = json.dumps({
            "relevance": 9,
            "clarity": 8,
            "answer_quality": 9,
            "educational_value": 7,
            "overall": 8.25,
            "feedback": "Add a bit more context."
        })

        with patch.object(CriticAgent, "_call", return_value=mock_response):
            scores = agent.evaluate(
                pair={"question": "What is Python?", "answer": "A programming language."},
                context="Python is a widely used high-level programming language."
            )

        assert scores["relevance"] == 9.0
        assert scores["clarity"] == 8.0
        assert scores["answer_quality"] == 9.0
        assert scores["educational_value"] == 7.0
        assert scores["overall"] == 8.25
        assert scores["feedback"] == "Add a bit more context."

    def test_evaluate_clamping(self, config):
        agent = CriticAgent(config)
        # Check clamping to 1-10 range
        mock_response = json.dumps({
            "relevance": 12,
            "clarity": -3,
            "answer_quality": 8,
            "educational_value": 8,
            "overall": 6.25,
            "feedback": "Fine."
        })

        with patch.object(CriticAgent, "_call", return_value=mock_response):
            scores = agent.evaluate(
                pair={"question": "Q", "answer": "A"},
                context="Ctx"
            )

        assert scores["relevance"] == 10.0
        assert scores["clarity"] == 1.0
        assert scores["overall"] == 6.25

    def test_evaluate_failure_fallback(self, config):
        agent = CriticAgent(config)

        # Non-JSON response
        with patch.object(CriticAgent, "_call", return_value="Failed response"):
            scores = agent.evaluate(
                pair={"question": "Q", "answer": "A"},
                context="Ctx"
            )

        assert scores["overall"] == 5.0
        assert "Could not parse" in scores["feedback"]


class TestRefinerAgent:
    def test_improve_success(self, config):
        agent = RefinerAgent(config)
        original_pair = {
            "question": "What is Python?",
            "answer": "A language.",
            "type": "factual",
            "chunk_id": 1
        }
        mock_response = json.dumps({
            "question": "What is Python in programming?",
            "answer": "Python is a high-level general-purpose programming language."
        })

        with patch.object(RefinerAgent, "_call", return_value=mock_response):
            improved = agent.improve(
                pair=original_pair,
                context="Python is a widely used high-level programming language.",
                scores={"overall": 5.0, "feedback": "Answer is too short."}
            )

        assert improved["question"] == "What is Python in programming?"
        assert improved["answer"] == "Python is a high-level general-purpose programming language."
        assert improved["type"] == "factual"
        assert improved["chunk_id"] == 1

    def test_improve_failure_fallback(self, config):
        agent = RefinerAgent(config)
        original_pair = {
            "question": "What is Python?",
            "answer": "A language.",
            "type": "factual",
            "chunk_id": 1
        }

        # Empty response
        with patch.object(RefinerAgent, "_call", return_value=None):
            improved = agent.improve(original_pair, "Ctx", {})
        assert improved == original_pair

        # Parse error
        with patch.object(RefinerAgent, "_call", return_value="Malformed"):
            improved = agent.improve(original_pair, "Ctx", {})
        assert improved == original_pair


class TestMultiAgentOrchestrator:
    def test_orchestrator_flow(self, tmp_config, sample_chunks):
        # Configure thresholds
        if "agents" not in tmp_config:
            tmp_config["agents"] = {}
        tmp_config["agents"]["quality_threshold"] = 7.0
        tmp_config["agents"]["max_refine_iterations"] = 2
        tmp_config["agents"]["max_qa_pairs"] = 5
        tmp_config["agents"]["question_types"] = ["factual"]

        orchestrator = MultiAgentOrchestrator(tmp_config)

        # 1. Mock GeneratorAgent.generate
        # Returns one raw pair
        gen_mock = MagicMock(return_value=[
            {"question": "What is a transformer?", "answer": "A deep learning model."}
        ])

        # 2. Mock CriticAgent.evaluate
        # Let's say it returns a low score first (6.0), then a high score (8.0) after refinement
        critic_scores = [
            # First critique
            {
                "relevance": 6.0, "clarity": 6.0, "answer_quality": 6.0, "educational_value": 6.0,
                "overall": 6.0, "feedback": "Make it more educational."
            },
            # Second critique (after refine)
            {
                "relevance": 8.0, "clarity": 8.0, "answer_quality": 8.0, "educational_value": 8.0,
                "overall": 8.0, "feedback": "Perfect."
            }
        ]
        critic_mock = MagicMock(side_effect=critic_scores)

        # 3. Mock RefinerAgent.improve
        # Returns an improved pair
        refine_mock = MagicMock(return_value={
            "question": "What is a transformer in deep learning?",
            "answer": "A transformer is a neural network architecture that uses self-attention."
        })

        with patch.object(GeneratorAgent, "generate", gen_mock), \
             patch.object(CriticAgent, "evaluate", critic_mock), \
             patch.object(RefinerAgent, "improve", refine_mock):
            
            # Run orchestrator on single chunk
            results = orchestrator.run(sample_chunks[:1])

        # Check results
        assert len(results) == 1
        assert results[0]["question"] == "What is a transformer in deep learning?"
        assert results[0]["quality_scores"]["overall"] == 8.0
        assert results[0]["refinement_iterations"] == 1
        assert results[0]["accepted"] is True

        # Check file outputs exist
        synth_dir = Path(tmp_config["paths"]["data_synthetic"])
        report_dir = Path(tmp_config["paths"]["reports"])
        
        assert (synth_dir / "synthetic_qa.json").exists()
        assert (synth_dir / "synthetic_qa.csv").exists()
        assert (report_dir / "agent_quality_report.json").exists()

    def test_orchestrator_discard_after_max_iterations(self, tmp_config, sample_chunks):
        if "agents" not in tmp_config:
            tmp_config["agents"] = {}
        tmp_config["agents"]["quality_threshold"] = 8.0
        tmp_config["agents"]["max_refine_iterations"] = 1
        tmp_config["agents"]["max_qa_pairs"] = 5
        tmp_config["agents"]["question_types"] = ["factual"]

        orchestrator = MultiAgentOrchestrator(tmp_config)

        # Generator returns a pair
        gen_mock = MagicMock(return_value=[
            {"question": "What is transformer?", "answer": "Model."}
        ])

        # Critic returns low score (5.0) both times
        critic_mock = MagicMock(return_value={
            "relevance": 5.0, "clarity": 5.0, "answer_quality": 5.0, "educational_value": 5.0,
            "overall": 5.0, "feedback": "Still bad."
        })

        refine_mock = MagicMock(return_value={
            "question": "What is transformer model?", "answer": "Model model."
        })

        with patch.object(GeneratorAgent, "generate", gen_mock), \
             patch.object(CriticAgent, "evaluate", critic_mock), \
             patch.object(RefinerAgent, "improve", refine_mock):
            
            results = orchestrator.run(sample_chunks[:1])

        # Should be empty since the pair was discarded
        assert len(results) == 0
