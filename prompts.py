"""
LLM Prompt Templates for Prompt Generation

These templates instruct Gemini to generate high-quality browser agent prompts
using real site context from the crawler.
"""
from __future__ import annotations

from typing import Any, Dict, List


def build_site_context_text(context: Dict[str, Any]) -> str:
    """Convert site context dict to readable text for the LLM."""
    lines = []
    
    lines.append(f"Website URL: {context.get('url', 'N/A')}")
    lines.append(f"Domain: {context.get('domain', 'N/A')}")
    lines.append(f"Site Type: {context.get('site_type', 'generic')}")
    
    if context.get('title'):
        lines.append(f"Title: {context['title']}")
    if context.get('description'):
        lines.append(f"Description: {context['description']}")
    
    # Vocabulary
    lines.append("")
    lines.append("=== Vocabulary ===")
    lines.append(f"Currency: {context.get('currency', '$')}")
    lines.append(f"Cart/Bag Word: {context.get('cart_word', 'cart')}")
    lines.append(f"Add Phrase: {context.get('add_phrase', 'add to cart')}")
    lines.append(f"Sign-in Word: {context.get('signin_word', 'sign in')}")
    
    # Features
    lines.append("")
    lines.append("=== Features Detected ===")
    features = []
    if context.get('has_search'):
        features.append("search")
    if context.get('has_checkout'):
        features.append("checkout")
    if context.get('has_account'):
        features.append("user accounts")
    if context.get('has_wishlist'):
        features.append("wishlist/favorites")
    if context.get('guest_checkout'):
        features.append("guest checkout")
    lines.append(f"Features: {', '.join(features) if features else 'None detected'}")
    
    # Navigation
    if context.get('main_sections'):
        lines.append("")
        lines.append("=== Main Navigation Sections ===")
        lines.append(", ".join(context['main_sections'][:12]))
    
    # Categories
    if context.get('categories'):
        lines.append("")
        lines.append("=== Categories ===")
        lines.append(", ".join(context['categories'][:12]))
    
    # Subcategories
    if context.get('subcategories'):
        lines.append("")
        lines.append("=== Subcategories ===")
        for cat, subs in list(context['subcategories'].items())[:5]:
            lines.append(f"  {cat}: {', '.join(subs[:5])}")
    
    # Filters
    if context.get('filter_types'):
        lines.append("")
        lines.append("=== Available Filters ===")
        lines.append(f"Filter types: {', '.join(context['filter_types'][:8])}")
        
        if context.get('filter_values'):
            for ftype, values in list(context['filter_values'].items())[:4]:
                lines.append(f"  {ftype} options: {', '.join(values[:8])}")
    
    # Sample products/topics
    if context.get('sample_products'):
        lines.append("")
        lines.append("=== Sample Products Found ===")
        for p in context['sample_products'][:10]:
            lines.append(f"  - {p}")
    
    if context.get('sample_topics'):
        lines.append("")
        lines.append("=== Sample Topics/Sections ===")
        for t in context['sample_topics'][:10]:
            lines.append(f"  - {t}")
    
    if context.get('search_suggestions'):
        lines.append("")
        lines.append("=== Search Suggestions ===")
        lines.append(", ".join(context['search_suggestions'][:8]))
    
    # Page types
    if context.get('page_types_found'):
        lines.append("")
        lines.append("=== Page Types Discovered ===")
        for ptype, count in context['page_types_found'].items():
            lines.append(f"  {ptype}: {count}")
    
    return "\n".join(lines)


def prompt_generation_template(
    site_context: Dict[str, Any],
    difficulty_levels: List[str],
    prompts_per_level: int,
    include_auth: bool = True,
    persona: Dict[str, str] = None
) -> str:
    """
    Generate the LLM prompt for creating browser agent prompts.
    
    This prompt includes the real site context from crawling,
    ensuring the LLM uses actual navigation terms and entities.
    """
    base_url = site_context.get('url', '')
    site_type = site_context.get('site_type', 'generic')
    context_text = build_site_context_text(site_context)
    
    difficulties_str = ", ".join(f'"{d}"' for d in difficulty_levels)
    total_prompts = len(difficulty_levels) * prompts_per_level
    
    # Persona info for forms
    persona_text = ""
    if persona:
        persona_text = f"""
For any prompts that involve filling forms (checkout, contact, etc.), use this persona:
- Name: {persona.get('name', 'Jane Doe')}
- Street: {persona.get('street', '123 Example Street')}
- City: {persona.get('city', 'Berlin')}
- ZIP: {persona.get('zip', '12345')}
- Country: {persona.get('country', 'Germany')}
- Phone: {persona.get('phone', '+49 170 1234567')}
"""
    
    auth_instruction = ""
    if include_auth:
        auth_instruction = """
For prompts that require authentication (marked with requires_credentials=true), include a step like:
"sign in using the provided credentials" or "log in with the provided credentials"
Do NOT embed actual usernames or passwords in the prompts.
"""
    else:
        auth_instruction = """
Do NOT include any prompts that require signing in or logging in.
All prompts must be executable without authentication.
"""
    
    return f"""You are an expert prompt engineer for browser automation agents. Your task is to generate high-quality, realistic prompts that an AI browser agent can execute.

CRITICAL: Use ONLY the real information from the Crawled Site Context below. Do NOT invent categories, products, filters, or navigation items that are not mentioned in the context.

=== Crawled Site Context ===
{context_text}

=== Your Task ===
Generate {total_prompts} concrete prompts for browser automation testing.

For each difficulty level [{difficulties_str}], generate exactly {prompts_per_level} prompts.

{persona_text}
{auth_instruction}

=== Difficulty Guidelines ===
- "trivial": 1-2 simple steps. Example: Open the site and find the contact page.
- "easy": 2-3 steps, maybe one simple action. Example: Search for X and open the first result.
- "fair": Multiple steps with some conditions. Example: Search for X, apply a filter, open a product.
- "hard": Longer flows with several conditions or navigation. Example: Find a category, apply multiple filters, compare products.
- "complex": Multi-step scenarios combining search, filters, cart, checkout (stopping before payment). Example: Find a product meeting specific criteria, add to cart, proceed to checkout, fill address.

=== Format Rules ===
1. Every prompt MUST start with: "Open {base_url} and ..."
2. Use action verbs: navigate, find, search, open, click, select, add, scroll, etc.
3. Be specific: Use actual category names, filter values, and navigation items from the context.
4. Be unambiguous: The agent should know exactly what success looks like.
5. Use the site's vocabulary: "{site_context.get('cart_word', 'cart')}" not "cart" if different, "{site_context.get('currency', '$')}" for prices.

=== Safety Rules ===
- Prompts can navigate through checkout flows but MUST stop before finalizing transactions.
- Acceptable endings: "proceed to checkout", "review the order", "reach the payment page"
- FORBIDDEN: "complete the purchase", "confirm the order", "submit payment", "place the order"

=== Output Format ===
Return ONLY a valid JSON object with this exact structure:

{{
  "prompts": [
    {{
      "difficulty": "trivial | easy | fair | hard | complex",
      "title": "short human-readable title",
      "prompt": "Open {base_url} and ...",
      "entities_used": ["list of specific items from context used in this prompt"],
      "requires_credentials": false
    }}
  ]
}}

Rules:
- Generate exactly {total_prompts} prompts total ({prompts_per_level} per difficulty level).
- Each prompt must be distinct and test different functionality.
- The "entities_used" field should list specific items from the context (categories, products, filters) used in that prompt.
- Output raw JSON only, no markdown code fences or explanations.
"""


def simple_fallback_prompt(
    url: str,
    site_type: str,
    difficulty_levels: List[str],
    prompts_per_level: int
) -> str:
    """
    Fallback prompt when crawling fails or returns minimal data.
    """
    total = len(difficulty_levels) * prompts_per_level
    difficulties_str = ", ".join(f'"{d}"' for d in difficulty_levels)
    
    return f"""You are an expert prompt engineer for browser automation agents.

Website: {url}
Detected Site Type: {site_type}

Generate {total} diverse prompts for testing browser automation on this website.
For each difficulty level [{difficulties_str}], generate exactly {prompts_per_level} prompts.

Difficulty Guidelines:
- "trivial": 1-2 simple navigation steps
- "easy": 2-3 steps with one action
- "fair": Multiple steps with conditions/filters
- "hard": Longer flows with multiple conditions
- "complex": Multi-step scenarios (stop before any payment/final confirmation)

Rules:
1. All prompts must start with "Open {url} and ..."
2. Be specific and unambiguous about success criteria
3. For checkout flows, stop at review/payment page (never finalize transactions)

Return ONLY valid JSON:
{{
  "prompts": [
    {{
      "difficulty": "trivial | easy | fair | hard | complex",
      "title": "short title",
      "prompt": "Open {url} and ...",
      "entities_used": [],
      "requires_credentials": false
    }}
  ]
}}
"""
