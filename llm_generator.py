"""
LLM Generator Module

Wraps Google Gemini API for prompt generation using crawled site context.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GENAI_OK = True
except ImportError:
    GENAI_OK = False
    genai = None

from prompts import prompt_generation_template, simple_fallback_prompt


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "gemini-2.5-flash"
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro",
    "gemini-3-ultra",
]

DEFAULT_DIFFICULTY_LEVELS = ["trivial", "easy", "fair", "hard", "complex"]


# =============================================================================
# Token/Cost Tracking
# =============================================================================

@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    
    def add(self, response_metadata):
        if hasattr(response_metadata, "prompt_token_count"):
            self.prompt_tokens += response_metadata.prompt_token_count
        if hasattr(response_metadata, "candidates_token_count"):
            self.completion_tokens += response_metadata.candidates_token_count
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.calls += 1


# =============================================================================
# Gemini Client
# =============================================================================

class GeminiGenerator:
    """
    Gemini API wrapper for generating browser agent prompts.
    
    Features:
    - JSON-only responses with retry logic
    - Token usage tracking
    - Configurable model selection
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        if not GENAI_OK:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        # Load API key from env if not provided
        if not api_key:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided. Set it in .env or pass as argument.")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.7,
                "max_output_tokens": 8192,
            }
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.usage = UsageStats()
    
    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences from response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate JSON response from Gemini.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Parsed JSON as dict
        """
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.model.generate_content(prompt)
                
                # Track usage
                if hasattr(response, "usage_metadata"):
                    self.usage.add(response.usage_metadata)
                
                raw_text = getattr(response, "text", "") or ""
                cleaned = self._strip_code_fences(raw_text)
                
                return json.loads(cleaned)
                
            except json.JSONDecodeError as e:
                last_error = e
                # Sometimes Gemini returns partial JSON, try to salvage
                if attempt >= self.max_retries:
                    raise ValueError(f"Failed to parse JSON response: {str(e)[:100]}")
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    raise
            
            time.sleep(self.retry_delay * attempt)
        
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected error in generate_json")
    
    def generate_prompts(
        self,
        site_context: Dict[str, Any],
        difficulty_levels: List[str] = None,
        prompts_per_level: int = 3,
        include_auth: bool = True,
        persona: Dict[str, str] = None,
        progress_callback = None
    ) -> List[Dict[str, Any]]:
        """
        Generate browser agent prompts using crawled site context.
        
        Args:
            site_context: Dict from SiteContext.asdict() or similar
            difficulty_levels: List of difficulty levels
            prompts_per_level: How many prompts per level
            include_auth: Whether to include auth-requiring prompts
            persona: Persona dict for form filling
            progress_callback: Optional callback(message)
            
        Returns:
            List of prompt dicts with keys: difficulty, title, prompt, entities_used, requires_credentials
        """
        if difficulty_levels is None:
            difficulty_levels = DEFAULT_DIFFICULTY_LEVELS
        
        if progress_callback:
            progress_callback("Preparing prompt for LLM...")
        
        # Check if we have enough context
        has_context = bool(
            site_context.get('main_sections') or
            site_context.get('categories') or
            site_context.get('sample_products') or
            site_context.get('filter_types')
        )
        
        if has_context:
            llm_prompt = prompt_generation_template(
                site_context=site_context,
                difficulty_levels=difficulty_levels,
                prompts_per_level=prompts_per_level,
                include_auth=include_auth,
                persona=persona
            )
        else:
            # Fallback if crawling returned minimal data
            if progress_callback:
                progress_callback("Limited crawl data, using fallback prompt...")
            llm_prompt = simple_fallback_prompt(
                url=site_context.get('url', ''),
                site_type=site_context.get('site_type', 'generic'),
                difficulty_levels=difficulty_levels,
                prompts_per_level=prompts_per_level
            )
        
        if progress_callback:
            progress_callback(f"Calling {self.model_name}...")
        
        try:
            result = self.generate_json(llm_prompt)
            prompts = result.get("prompts", [])
            
            if progress_callback:
                progress_callback(f"Generated {len(prompts)} prompts")
            
            return prompts
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"LLM error: {e}")
            raise


# =============================================================================
# Convenience Function
# =============================================================================

def generate_prompts_from_context(
    site_context: Dict[str, Any],
    api_key: str = None,
    model_name: str = DEFAULT_MODEL,
    difficulty_levels: List[str] = None,
    prompts_per_level: int = 3,
    include_auth: bool = True,
    persona: Dict[str, str] = None,
    progress_callback = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate prompts from site context.
    
    Args:
        site_context: Dict with site information (from crawler)
        api_key: Gemini API key (or set GEMINI_API_KEY env var)
        model_name: Which Gemini model to use
        difficulty_levels: List of difficulty levels
        prompts_per_level: Prompts per difficulty level
        include_auth: Include prompts requiring auth
        persona: Persona for form filling
        progress_callback: Progress updates
        
    Returns:
        List of prompt dicts
    """
    generator = GeminiGenerator(api_key=api_key, model_name=model_name)
    return generator.generate_prompts(
        site_context=site_context,
        difficulty_levels=difficulty_levels,
        prompts_per_level=prompts_per_level,
        include_auth=include_auth,
        persona=persona,
        progress_callback=progress_callback
    )
