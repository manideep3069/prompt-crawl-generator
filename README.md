# Hybrid Prompt Generator

This tool combines **intelligent web crawling** with **LLM-powered prompt generation** to create high-quality evaluation prompts for browser automation agents.

## Setup

```bash
cd prompt_generator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Configuration

Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

```bash
streamlit run app.py
```

## Features

- **Intelligent Crawling**: Uses Playwright to deeply analyze website structure
- **Smart Entity Extraction**: Finds real navigation terms, products, filters
- **LLM-Powered Generation**: Gemini creates diverse, realistic prompts
- **Difficulty Levels**: trivial, easy, fair, hard, complex
- **Safety Rules**: Prompts stop before payment/transactions
