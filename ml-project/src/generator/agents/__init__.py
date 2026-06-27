"""
src/generator/agents/__init__.py

Multi-Agent Synthetic Data System.

Exposes the main entry point (MultiAgentOrchestrator) and individual agents
so they can be imported directly or used in tests.
"""
from .orchestrator import MultiAgentOrchestrator
from .generator_agent import GeneratorAgent
from .critic_agent import CriticAgent
from .refiner_agent import RefinerAgent

__all__ = [
    "MultiAgentOrchestrator",
    "GeneratorAgent",
    "CriticAgent",
    "RefinerAgent",
]
