"""
Hybrid Prompt Generator - Streamlit App

Combines intelligent web crawling with LLM-powered prompt generation
for high-quality browser agent evaluation prompts.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

# Local imports
from crawler import crawl_site, SiteContext, CRAWL4AI_OK
from llm_generator import (
    GeminiGenerator,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_DIFFICULTY_LEVELS,
    GENAI_OK
)


# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Hybrid Prompt Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .crawl-stat {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
    }
    .entity-tag {
        background-color: #e3f2fd;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px;
        display: inline-block;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar Configuration
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Crawl settings
    st.subheader("🌐 Crawl Settings")
    max_pages = st.slider("Max pages to crawl", 5, 50, 20)
    max_depth = st.slider("Max crawl depth", 1, 4, 2)
    headless = st.checkbox("Headless browser", value=True)
    timeout_ms = st.slider("Page timeout (ms)", 5000, 30000, 15000, 1000)
    
    if not CRAWL4AI_OK:
        st.warning("crawl4ai not installed. Run:\n`pip install crawl4ai`")
    
    st.divider()
    
    # LLM settings
    st.subheader("🤖 LLM Settings")
    
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Or set GEMINI_API_KEY in .env file"
    )
    
    model_name = st.selectbox("Model", AVAILABLE_MODELS, index=0)
    
    if not GENAI_OK:
        st.warning("google-generativeai not installed. Run:\n`pip install google-generativeai`")
    
    st.divider()
    
    # Prompt settings
    st.subheader("📝 Prompt Settings")
    
    difficulty_options = st.multiselect(
        "Difficulty levels",
        DEFAULT_DIFFICULTY_LEVELS,
        default=DEFAULT_DIFFICULTY_LEVELS
    )
    
    prompts_per_level = st.slider("Prompts per level", 1, 5, 3)
    include_auth = st.checkbox("Include auth-requiring prompts", value=False)
    
    st.divider()
    
    # Persona settings
    st.subheader("👤 Persona (for forms)")
    persona_name = st.text_input("Name", "Jane Doe")
    persona_street = st.text_input("Street", "Example Street 123")
    persona_zip = st.text_input("ZIP", "12345")
    persona_city = st.text_input("City", "Berlin")
    persona_country = st.text_input("Country", "Germany")
    persona_phone = st.text_input("Phone", "+49 170 1234567")

persona = {
    "name": persona_name,
    "street": persona_street,
    "zip": persona_zip,
    "city": persona_city,
    "country": persona_country,
    "phone": persona_phone
}


# =============================================================================
# Main UI
# =============================================================================

st.title("🔍 Hybrid Prompt Generator")
st.markdown("Combines **intelligent web crawling** with **LLM-powered generation** for high-quality browser agent prompts.")

# URL Input
url_input = st.text_input(
    "🌐 Website URL",
    placeholder="https://example.com",
    help="Enter any website URL to analyze and generate prompts"
)

# Two-column layout for controls
col1, col2 = st.columns([1, 1])

with col1:
    crawl_button = st.button("🕷️ Step 1: Crawl Website", type="secondary", use_container_width=True)

with col2:
    generate_button = st.button("✨ Step 2: Generate Prompts", type="primary", use_container_width=True)


# =============================================================================
# Session State
# =============================================================================

if "site_context" not in st.session_state:
    st.session_state.site_context = None
if "prompts" not in st.session_state:
    st.session_state.prompts = None
if "crawl_logs" not in st.session_state:
    st.session_state.crawl_logs = []


# =============================================================================
# Crawl Action
# =============================================================================

if crawl_button:
    if not url_input:
        st.error("Please enter a URL")
    elif not CRAWL4AI_OK:
        st.error("crawl4ai not installed")
    else:
        st.session_state.crawl_logs = []
        st.session_state.prompts = None
        
        progress_container = st.empty()
        log_container = st.empty()
        
        def update_progress(msg: str):
            st.session_state.crawl_logs.append(msg)
            with log_container:
                st.text("\n".join(st.session_state.crawl_logs[-10:]))
        
        with st.spinner("Crawling website with Crawl4AI..."):
            try:
                from crawler import Crawl4AICrawler
                crawler = Crawl4AICrawler(
                    max_pages=max_pages,
                    timeout_ms=timeout_ms,
                    headless=headless
                )
                context = crawler.crawl(url_input, progress_callback=update_progress)
                st.session_state.site_context = asdict(context)
                st.success(f"✅ Crawl complete! Analyzed {context.pages_crawled} pages.")
            except Exception as e:
                st.error(f"Crawl failed: {e}")


# =============================================================================
# Display Crawl Results
# =============================================================================

if st.session_state.site_context:
    ctx = st.session_state.site_context
    
    st.divider()
    st.subheader("📊 Crawl Results")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pages Crawled", ctx.get("pages_crawled", 0))
    with col2:
        st.metric("Site Type", ctx.get("site_type", "generic").title())
    with col3:
        st.metric("Currency", ctx.get("currency", "$"))
    with col4:
        st.metric("Cart Word", ctx.get("cart_word", "cart").title())
    
    # Tabs for different data categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📑 Navigation", "🏷️ Categories", "🔧 Filters", "📦 Products/Topics", "📋 Raw Data"
    ])
    
    with tab1:
        st.markdown("**Main Navigation Sections:**")
        sections = ctx.get("main_sections", [])
        if sections:
            cols = st.columns(4)
            for i, section in enumerate(sections):
                with cols[i % 4]:
                    st.markdown(f"• {section}")
        else:
            st.info("No navigation sections detected")
        
        st.markdown("**Features Detected:**")
        features = []
        if ctx.get("has_search"):
            features.append("🔍 Search")
        if ctx.get("has_checkout"):
            features.append("🛒 Checkout")
        if ctx.get("has_account"):
            features.append("👤 User Accounts")
        if ctx.get("has_wishlist"):
            features.append("❤️ Wishlist")
        if ctx.get("guest_checkout"):
            features.append("🚶 Guest Checkout")
        st.markdown(" | ".join(features) if features else "None detected")
    
    with tab2:
        categories = ctx.get("categories", [])
        subcategories = ctx.get("subcategories", {})
        
        if categories:
            st.markdown("**Categories Found:**")
            for cat in categories:
                if cat in subcategories:
                    with st.expander(f"📁 {cat}"):
                        for sub in subcategories[cat]:
                            st.markdown(f"  └ {sub}")
                else:
                    st.markdown(f"• {cat}")
        else:
            st.info("No categories detected")
    
    with tab3:
        filter_types = ctx.get("filter_types", [])
        filter_values = ctx.get("filter_values", {})
        
        if filter_types:
            st.markdown("**Filter Types:**")
            for ftype in filter_types:
                values = filter_values.get(ftype, [])
                with st.expander(f"🔧 {ftype.title()}"):
                    if values:
                        st.markdown(", ".join(values[:15]))
                    else:
                        st.markdown("_No values extracted_")
        else:
            st.info("No filters detected (try crawling a category/listing page)")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sample Products:**")
            products = ctx.get("sample_products", [])
            if products:
                for p in products[:15]:
                    st.markdown(f"• {p[:60]}{'...' if len(p) > 60 else ''}")
            else:
                st.info("No products detected")
        
        with col2:
            st.markdown("**Topics/Sections:**")
            topics = ctx.get("sample_topics", [])
            if topics:
                for t in topics[:15]:
                    st.markdown(f"• {t[:60]}{'...' if len(t) > 60 else ''}")
            else:
                st.info("No topics detected")
        
        suggestions = ctx.get("search_suggestions", [])
        if suggestions:
            st.markdown("**Search Suggestions:**")
            st.markdown(", ".join(suggestions[:10]))
    
    with tab5:
        st.markdown("**Page Types Discovered:**")
        page_types = ctx.get("page_types_found", {})
        if page_types:
            df_types = pd.DataFrame([
                {"Page Type": k, "Count": v}
                for k, v in page_types.items()
            ])
            st.dataframe(df_types, use_container_width=True, hide_index=True)
        
        with st.expander("View Full JSON"):
            st.json(ctx)
        
        st.markdown("**Crawl Log:**")
        for log in ctx.get("crawl_notes", [])[-20:]:
            st.text(log)


# =============================================================================
# Generate Prompts Action
# =============================================================================

if generate_button:
    if not st.session_state.site_context:
        st.error("Please crawl a website first (Step 1)")
    elif not api_key:
        st.error("Please enter your Gemini API key in the sidebar")
    elif not GENAI_OK:
        st.error("google-generativeai not installed")
    elif not difficulty_options:
        st.error("Please select at least one difficulty level")
    else:
        with st.spinner("Generating prompts with Gemini..."):
            try:
                generator = GeminiGenerator(api_key=api_key, model_name=model_name)
                prompts = generator.generate_prompts(
                    site_context=st.session_state.site_context,
                    difficulty_levels=difficulty_options,
                    prompts_per_level=prompts_per_level,
                    include_auth=include_auth,
                    persona=persona
                )
                st.session_state.prompts = prompts
                
                st.success(f"✅ Generated {len(prompts)} prompts!")
                st.info(f"Tokens used: {generator.usage.total_tokens}")
                
            except Exception as e:
                st.error(f"Generation failed: {e}")


# =============================================================================
# Display Generated Prompts
# =============================================================================

if st.session_state.prompts:
    prompts = st.session_state.prompts
    
    st.divider()
    st.subheader("✨ Generated Prompts")
    
    # Group by difficulty
    by_difficulty = {}
    for p in prompts:
        d = p.get("difficulty", "unknown")
        if d not in by_difficulty:
            by_difficulty[d] = []
        by_difficulty[d].append(p)
    
    # Display in tabs
    if by_difficulty:
        tabs = st.tabs([f"{d.title()} ({len(ps)})" for d, ps in by_difficulty.items()])
        
        for tab, (difficulty, ps) in zip(tabs, by_difficulty.items()):
            with tab:
                for i, p in enumerate(ps):
                    with st.expander(f"**{p.get('title', f'Prompt {i+1}')}**", expanded=i == 0):
                        st.code(p.get("prompt", ""), language=None)
                        
                        entities = p.get("entities_used", [])
                        if entities:
                            st.markdown("**Entities used:** " + ", ".join(entities))
                        
                        if p.get("requires_credentials"):
                            st.warning("🔐 Requires credentials")
    
    # Export options
    st.divider()
    st.subheader("📥 Export")
    
    col1, col2, col3 = st.columns(3)
    
    # Prepare data
    df = pd.DataFrame(prompts)
    bulk_text = "\n\n".join([
        f"[{p.get('difficulty', 'unknown').upper()}] {p.get('prompt', '')}"
        for p in prompts
    ])
    
    with col1:
        st.download_button(
            "📄 Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="prompts.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            "📝 Download TXT",
            data=bulk_text.encode("utf-8"),
            file_name="prompts.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        st.download_button(
            "🔧 Download JSON",
            data=json.dumps(prompts, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="prompts.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Bulk copy area
    with st.expander("📋 Bulk Copy (All Prompts)"):
        st.text_area("All prompts", value=bulk_text, height=300)


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em;">
    Hybrid Prompt Generator • Crawl + LLM = Quality Prompts
</div>
""", unsafe_allow_html=True)
