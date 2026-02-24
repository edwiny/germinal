"""
Security validation layer for tool outputs before they're sent to the LLM.

This module provides a modular, extensible system for validating and sanitizing
tool outputs to prevent security issues like sensitive data leakage and prompt
injection attacks.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

logger = logging.getLogger("security")


class OutputValidator(ABC):
    """
    Abstract base class for output validators.

    Validators can modify the result dict, add metadata, or raise exceptions
    to block potentially dangerous outputs.
    """

    @abstractmethod
    def validate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and potentially modify a tool result.

        Args:
            result: The raw tool execution result dict

        Returns:
            The validated (and potentially modified) result dict

        Raises:
            SecurityException: If the result should be blocked
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the validator's name for logging and configuration."""
        pass


class SecurityException(Exception):
    """Raised when a security validator blocks a tool output."""
    pass


class SensitiveDataMasker(OutputValidator):
    """
    Masks sensitive data patterns in tool outputs.

    Currently masks:
    - API keys (patterns like 'sk-', 'pk-', 'Bearer ', etc.)
    - Generic tokens and secrets
    - Password fields
    - Private keys
    """

    # Common sensitive data patterns
    SENSITIVE_PATTERNS = [
        # API keys and tokens
        (r'\b(sk-[a-zA-Z0-9_-]{10,})\b', '[API_KEY_MASKED]'),
        (r'\b(pk_[a-zA-Z0-9_-]{10,})\b', '[API_KEY_MASKED]'),
        (r'\b(Bearer\s+[a-zA-Z0-9_.-]{10,})\b', '[BEARER_TOKEN_MASKED]'),
        (r'\b(Authorization:\s*[a-zA-Z0-9_.-]{10,})\b', '[AUTH_HEADER_MASKED]'),

        # Generic secrets
        (r'\b(secret[_-]?[a-zA-Z0-9_-]{5,})\b', '[SECRET_MASKED]'),
        (r'\b(token[_-]?[a-zA-Z0-9_-]{5,})\b', '[TOKEN_MASKED]'),
        (r'\b(password[_-]?[a-zA-Z0-9_-]{5,})\b', '[PASSWORD_MASKED]'),

        # Private keys (basic pattern)
        (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----.*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
         '[PRIVATE_KEY_MASKED]'),
    ]

    def validate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in the result."""
        masked_result = self._mask_dict(result)
        if masked_result != result:
            logger.warning("Sensitive data masked in tool output")
        return masked_result

    def _mask_dict(self, data: Any) -> Any:
        """Recursively mask sensitive data in nested dict/list structures."""
        if isinstance(data, dict):
            return {k: self._mask_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._mask_dict(item) for item in data]
        elif isinstance(data, str):
            masked = data
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE | re.DOTALL)
            return masked
        else:
            return data

    @property
    def name(self) -> str:
        return "sensitive_data_masker"


class PromptInjectionDetector(OutputValidator):
    """
    Basic detection of potential prompt injection attacks in tool outputs.

    This is a simple pattern-based detector that looks for common injection
    patterns. It's not foolproof but provides a basic defense layer.
    """

    # Patterns that might indicate prompt injection attempts
    INJECTION_PATTERNS = [
        # Direct system prompt overrides
        r'(?i)(system\s+prompt|you\s+are\s+now|ignore\s+previous|forget\s+your)',
        # Role changes
        r'(?i)(act\s+as|role\s*play|pretend\s+to\s+be)',
        # Instruction overrides
        r'(?i)(override|disregard|ignore.*instruction)',
        # Dangerous commands
        r'(?i)(execute.*command|run.*script|delete.*file|format.*disk)',
    ]

    def validate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential prompt injection patterns."""
        text_content = self._extract_text_content(result)

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_content, re.IGNORECASE):
                logger.warning(f"Potential prompt injection detected: {pattern}")
                # For now, we log but don't block - this could be made configurable
                # In a production system, you might want to:
                # raise SecurityException("Potential prompt injection detected")

        return result

    def _extract_text_content(self, data: Any) -> str:
        """Extract all string content from nested structures for analysis."""
        if isinstance(data, dict):
            return ' '.join(str(v) for v in data.values() if isinstance(v, (str, int, float)))
        elif isinstance(data, list):
            return ' '.join(str(item) for item in data if isinstance(item, (str, int, float)))
        elif isinstance(data, str):
            return data
        else:
            return str(data)

    @property
    def name(self) -> str:
        return "prompt_injection_detector"


class ValidationPipeline:
    """
    Orchestrates multiple validators in sequence.

    Validators are applied in order, with each receiving the output of the previous.
    """

    def __init__(self, validators: Optional[List[OutputValidator]] = None):
        self.validators = validators or []
        self.enabled = True

    def add_validator(self, validator: OutputValidator) -> None:
        """Add a validator to the pipeline."""
        self.validators.append(validator)

    def validate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all validators on the result.

        Returns the final validated result, or raises SecurityException if
        any validator blocks the output.
        """
        if not self.enabled:
            return result

        validated_result = result
        for validator in self.validators:
            try:
                validated_result = validator.validate(validated_result)
                logger.debug(f"Validator {validator.name} passed")
            except SecurityException as e:
                logger.error(f"Validator {validator.name} blocked output: {e}")
                raise
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {e}")
                # Continue with other validators even if one fails
                continue

        return validated_result

    def enable(self) -> None:
        """Enable the validation pipeline."""
        self.enabled = True

    def disable(self) -> None:
        """Disable the validation pipeline."""
        self.enabled = False


# Global validation pipeline instance - initialized with defaults
_default_pipeline = ValidationPipeline([
    SensitiveDataMasker(),
    PromptInjectionDetector(),
])


def get_default_pipeline() -> ValidationPipeline:
    """Get the default validation pipeline."""
    return _default_pipeline


def create_pipeline_from_config(config: Dict[str, Any]) -> ValidationPipeline:
    """
    Create a validation pipeline from configuration.

    Args:
        config: Security configuration dict with 'enabled' and 'validators' keys

    Returns:
        Configured ValidationPipeline instance
    """
    pipeline = ValidationPipeline()

    if not config.get('enabled', True):
        pipeline.disable()
        return pipeline

    validator_names = config.get('validators', ['sensitive_data_masker', 'prompt_injection_detector'])

    # Map validator names to classes
    validator_classes = {
        'sensitive_data_masker': SensitiveDataMasker,
        'prompt_injection_detector': PromptInjectionDetector,
    }

    for name in validator_names:
        if name in validator_classes:
            pipeline.add_validator(validator_classes[name]())
        else:
            logger.warning(f"Unknown validator '{name}' in security config, skipping")

    return pipeline


def validate_tool_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a tool output using the default pipeline.

    This is the main entry point for validating tool outputs.
    """
    return _default_pipeline.validate(result)
