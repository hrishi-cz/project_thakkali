"""
Core Intelligence Layer - Single Source of Truth for APEX Pipeline.
"""

from core.execution_context import ExecutionContext, DatasetProfile, validate_context

__all__ = ['ExecutionContext', 'DatasetProfile', 'validate_context']
