"""
Web Crawler using Crawl4AI ONLY

Uses crawl4ai's native AsyncWebCrawler for:
- LLM-optimized markdown output
- Async multi-page crawling
- Built-in content extraction
- Link discovery
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse
from collections import defaultdict

from bs4 import BeautifulSoup
import tldextract

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_OK = True
except ImportError:
    CRAWL4AI_OK = False
    AsyncWebCrawler = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SiteContext:
    """Complete site context for LLM prompt generation."""
    url: str
    domain: str = ""
    site_type: str = "generic"
    
    title: str = ""
    description: str = ""
    
    currency: str = "$"
    cart_word: str = "cart"
    add_phrase: str = "add to cart"
    signin_word: str = "sign in"
    
    main_sections: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    subcategories: Dict[str, List[str]] = field(default_factory=dict)
    
    filter_types: List[str] = field(default_factory=list)
    filter_values: Dict[str, List[str]] = field(default_factory=dict)
    search_suggestions: List[str] = field(default_factory=list)
    
    sample_products: List[str] = field(default_factory=list)
    sample_services: List[str] = field(default_factory=list)
    sample_topics: List[str] = field(default_factory=list)
    
    internal_links: List[str] = field(default_factory=list)
    
    pages_crawled: int = 0
    page_types_found: Dict[str, int] = field(default_factory=dict)
    
    has_search: bool = False
    has_checkout: bool = False
    has_account: bool = False
    has_wishlist: bool = False
    guest_checkout: bool = False
    
    markdown_content: str = ""
    crawl_notes: List[str] = field(default_factory=list)


# =============================================================================
# Utilities
# =============================================================================

def ensure_http(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")

def get_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def same_site(base: str, u: str) -> bool:
    return get_domain(base) == get_domain(u)

STOP_NAV_WORDS = {
    "home", "menu", "search", "account", "profile", "sign in", "log in", "login",
    "cart", "bag", "basket", "wishlist", "favorites", "help", "support", "contact",
    "about", "privacy", "terms", "cookie", "legal", "careers", "press", "blog"
}

PRIORITY_PATTERNS = [
    r"/category/", r"/c/", r"/collection/", r"/shop/",
    r"/products?/", r"/women", r"/men", r"/sale", r"/new",
]


# =============================================================================
# Content Analysis
# =============================================================================

def detect_site_type(url: str, text: str) -> str:
    h = urlparse(url).netloc.lower()
    text_lower = text.lower()
    
    if "github.com" in h or "gitlab" in h:
        return "devplatform"
    if "aws.amazon.com" in h or "cloud.google.com" in h:
        return "cloud"
    if any(k in h for k in ["docs.", "documentation."]):
        return "docs"
    
    ecom = sum(text_lower.count(k) for k in [
        "add to cart", "add to bag", "shopping", "checkout", "price", "buy"
    ])
    docs = sum(text_lower.count(k) for k in [
        "documentation", "api", "getting started", "tutorial"
    ])
    
    if ecom > docs and ecom > 3:
        return "ecommerce"
    if docs > ecom and docs > 3:
        return "docs"
    return "generic"


def detect_vocabulary(text: str) -> Dict[str, str]:
    text_lower = text.lower()
    vocab = {"currency": "$", "cart_word": "cart", "add_phrase": "add to cart", "signin_word": "sign in"}
    
    if "€" in text:
        vocab["currency"] = "€"
    elif "£" in text:
        vocab["currency"] = "£"
    
    if "add to bag" in text_lower:
        vocab["cart_word"] = "bag"
        vocab["add_phrase"] = "add to bag"
    elif "add to basket" in text_lower:
        vocab["cart_word"] = "basket"
        vocab["add_phrase"] = "add to basket"
    
    if "log in" in text_lower:
        vocab["signin_word"] = "log in"
    
    return vocab


def detect_features(text: str) -> Dict[str, bool]:
    text_lower = text.lower()
    return {
        "has_search": "search" in text_lower,
        "has_checkout": "checkout" in text_lower,
        "has_account": "account" in text_lower or "sign in" in text_lower,
        "has_wishlist": "wishlist" in text_lower or "favorites" in text_lower,
        "guest_checkout": "guest checkout" in text_lower or "continue as guest" in text_lower
    }


def extract_nav_sections(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    sections = []
    
    for nav in soup.select("nav, header, [role='navigation']"):
        for link in nav.select("a"):
            text = clean_text(link.get_text())
            if 2 < len(text) < 25 and text.lower() not in STOP_NAV_WORDS:
                if text not in sections:
                    sections.append(text)
    
    return sections[:20]


def extract_categories(html: str) -> tuple[List[str], Dict[str, List[str]]]:
    soup = BeautifulSoup(html, "lxml")
    categories = []
    subcategories = {}
    
    for container in soup.select("[class*='category'], [class*='menu'], [class*='dropdown'], [class*='nav-item']"):
        parent = None
        for heading in container.select("a, span, h2, h3, h4"):
            text = clean_text(heading.get_text())
            if text and 2 < len(text) < 25 and text.lower() not in STOP_NAV_WORDS:
                parent = text
                if parent not in categories:
                    categories.append(parent)
                break
        
        children = []
        for link in container.select("ul a, li a"):
            text = clean_text(link.get_text())
            if text and 2 < len(text) < 30 and text != parent:
                if text.lower() not in STOP_NAV_WORDS and text not in children:
                    children.append(text)
        
        if parent and children:
            subcategories[parent] = children[:10]
    
    return categories[:15], subcategories


def extract_filters(html: str) -> tuple[List[str], Dict[str, List[str]]]:
    soup = BeautifulSoup(html, "lxml")
    filter_types = []
    filter_values = {}
    
    filter_keywords = {
        "size": ["size", "größe"],
        "color": ["color", "colour", "farbe"],
        "brand": ["brand", "marke"],
        "price": ["price", "preis"],
    }
    
    for container in soup.select("[class*='filter'], [class*='facet'], [class*='refine']"):
        container_text = clean_text(container.get_text(" ")).lower()
        
        for filter_name, keywords in filter_keywords.items():
            if any(kw in container_text for kw in keywords):
                if filter_name not in filter_types:
                    filter_types.append(filter_name)
                
                values = []
                for el in container.select("input, label, a, button, li"):
                    val = clean_text(el.get_text() or el.get("value", ""))
                    if val and 1 < len(val) < 25:
                        if val.lower() not in {"all", "clear", "apply", "filter"}:
                            values.append(val)
                
                if values:
                    filter_values[filter_name] = list(dict.fromkeys(values))[:12]
    
    return filter_types, filter_values


def extract_products_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    products = []
    
    for product in soup.select("[class*='product'], [class*='item'], article, [data-testid*='product']"):
        for name_el in product.select("h2, h3, h4, [class*='name'], [class*='title']"):
            name = clean_text(name_el.get_text())
            if name and 5 < len(name) < 80 and name not in products:
                products.append(name)
                break
    
    return products[:25]


def extract_products_from_markdown(markdown: str) -> List[str]:
    products = []
    
    # Links with product-like text
    link_pattern = re.compile(r'\[([^\]]{10,60})\]\([^\)]+\)')
    for match in link_pattern.finditer(markdown):
        text = match.group(1).strip()
        if not any(w in text.lower() for w in ["click", "view", "see all", "read more"]):
            if text not in products:
                products.append(text)
    
    return products[:20]


def extract_internal_links(links_dict: Dict, base_url: str) -> List[str]:
    """Extract internal links from crawl4ai's links dict."""
    internal = []
    
    if not links_dict:
        return internal
    
    internal_links = links_dict.get("internal", {})
    if isinstance(internal_links, dict):
        for href in internal_links.keys():
            if same_site(base_url, href):
                if not re.search(r'\.(jpg|png|gif|css|js|pdf)(\?|$)', href, re.I):
                    internal.append(href)
    elif isinstance(internal_links, list):
        for item in internal_links:
            href = item if isinstance(item, str) else item.get("href", "")
            if href and same_site(base_url, href):
                if not re.search(r'\.(jpg|png|gif|css|js|pdf)(\?|$)', href, re.I):
                    internal.append(href)
    
    return internal[:50]


def classify_page_type(url: str) -> str:
    url_lower = url.lower()
    
    patterns = {
        "cart": ["/cart", "/basket", "/bag"],
        "checkout": ["/checkout", "/payment"],
        "product": ["/product/", "/item/", "/p/", "/dp/"],
        "category": ["/category/", "/c/", "/collection/", "/shop/", "/women", "/men"],
        "search": ["/search", "?q="],
        "account": ["/account", "/profile", "/login"],
        "docs": ["/docs", "/documentation", "/api"],
        "pricing": ["/pricing", "/plans"],
    }
    
    for ptype, keywords in patterns.items():
        if any(k in url_lower for k in keywords):
            return ptype
    
    return "general"


# =============================================================================
# Crawl4AI Crawler
# =============================================================================

class Crawl4AICrawler:
    """
    Multi-page crawler using only Crawl4AI.
    
    Features:
    - Crawls minimum 5 pages
    - LLM-optimized markdown output
    - Smart link prioritization
    - Content extraction from all pages
    """
    
    def __init__(
        self,
        max_pages: int = 15,
        min_pages: int = 5,
        headless: bool = True,
        timeout_ms: int = 30000
    ):
        self.max_pages = max(max_pages, min_pages)
        self.min_pages = min_pages
        self.headless = headless
        self.timeout_ms = timeout_ms
    
    def crawl(self, url: str, progress_callback=None) -> SiteContext:
        """Synchronous wrapper."""
        return asyncio.run(self._crawl_async(url, progress_callback))
    
    async def _crawl_async(self, url: str, progress_callback=None) -> SiteContext:
        if not CRAWL4AI_OK:
            raise RuntimeError("crawl4ai not installed. Run: pip install crawl4ai")
        
        url = ensure_http(url)
        domain = get_domain(url)
        
        ctx = SiteContext(url=url, domain=domain, page_types_found=defaultdict(int))
        
        def log(msg: str):
            ctx.crawl_notes.append(msg)
            if progress_callback:
                progress_callback(msg)
        
        log(f"Starting Crawl4AI crawl of {url}")
        log(f"Target: min {self.min_pages} pages, max {self.max_pages} pages")
        
        # Configure browser
        browser_config = BrowserConfig(
            headless=self.headless,
            viewport_width=1280,
            viewport_height=900,
        )
        
        # Crawler config
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=self.timeout_ms,
            wait_until="domcontentloaded",
        )
        
        visited: Set[str] = set()
        to_visit: List[str] = []
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # === Crawl homepage ===
            log("Phase 1: Crawling homepage...")
            
            try:
                result = await crawler.arun(url, config=crawler_config)
                
                if result.success:
                    ctx.pages_crawled += 1
                    ctx.page_types_found["homepage"] = 1
                    visited.add(url)
                    
                    # Extract metadata
                    if result.metadata:
                        ctx.title = result.metadata.get("title", "")
                        ctx.description = result.metadata.get("description", "")[:300]
                    
                    # Get text content
                    text = result.markdown or result.cleaned_html or ""
                    html = result.html or ""
                    ctx.markdown_content = text[:10000]
                    
                    # Detect site type and vocabulary
                    ctx.site_type = detect_site_type(url, text)
                    log(f"  Site type: {ctx.site_type}")
                    
                    vocab = detect_vocabulary(text)
                    ctx.currency = vocab["currency"]
                    ctx.cart_word = vocab["cart_word"]
                    ctx.add_phrase = vocab["add_phrase"]
                    ctx.signin_word = vocab["signin_word"]
                    
                    # Detect features
                    features = detect_features(text)
                    ctx.has_search = features["has_search"]
                    ctx.has_checkout = features["has_checkout"]
                    ctx.has_account = features["has_account"]
                    ctx.has_wishlist = features["has_wishlist"]
                    ctx.guest_checkout = features["guest_checkout"]
                    
                    # Extract navigation
                    ctx.main_sections = extract_nav_sections(html)
                    log(f"  Found {len(ctx.main_sections)} nav sections")
                    
                    # Extract categories
                    cats, subcats = extract_categories(html)
                    ctx.categories = cats
                    ctx.subcategories = subcats
                    
                    # Extract products
                    products = extract_products_from_html(html)
                    products += extract_products_from_markdown(text)
                    ctx.sample_products = list(dict.fromkeys(products))[:25]
                    
                    # Get internal links for Phase 2
                    if result.links:
                        to_visit = extract_internal_links(result.links, url)
                        log(f"  Found {len(to_visit)} internal links")
                else:
                    log(f"  Homepage failed: {result.error_message}")
            
            except Exception as e:
                log(f"  Homepage error: {e}")
            
            # === Crawl additional pages ===
            if to_visit:
                log(f"Phase 2: Crawling additional pages...")
                
                # Prioritize interesting pages
                priority = []
                normal = []
                for link in to_visit:
                    if link not in visited:
                        if any(re.search(p, link, re.I) for p in PRIORITY_PATTERNS):
                            priority.append(link)
                        else:
                            normal.append(link)
                
                queue = priority[:20] + normal[:10]
                
                for link in queue:
                    if ctx.pages_crawled >= self.max_pages:
                        break
                    if link in visited:
                        continue
                    
                    visited.add(link)
                    
                    try:
                        result = await crawler.arun(link, config=crawler_config)
                        
                        if result.success:
                            ctx.pages_crawled += 1
                            ptype = classify_page_type(link)
                            ctx.page_types_found[ptype] = ctx.page_types_found.get(ptype, 0) + 1
                            
                            log(f"  [{ctx.pages_crawled}] {ptype}: {link[:50]}...")
                            
                            html = result.html or ""
                            text = result.markdown or ""
                            
                            # Extract more content from this page
                            if ptype in ["category", "search"]:
                                ftypes, fvals = extract_filters(html)
                                for ft in ftypes:
                                    if ft not in ctx.filter_types:
                                        ctx.filter_types.append(ft)
                                for k, v in fvals.items():
                                    if k not in ctx.filter_values:
                                        ctx.filter_values[k] = []
                                    ctx.filter_values[k].extend([x for x in v if x not in ctx.filter_values[k]])
                            
                            # More products
                            products = extract_products_from_html(html)
                            products += extract_products_from_markdown(text)
                            for p in products:
                                if p not in ctx.sample_products and len(ctx.sample_products) < 30:
                                    ctx.sample_products.append(p)
                            
                            # More categories
                            more_cats, more_subs = extract_categories(html)
                            for c in more_cats:
                                if c not in ctx.categories:
                                    ctx.categories.append(c)
                            ctx.subcategories.update(more_subs)
                            
                            # More links
                            if result.links:
                                new_links = extract_internal_links(result.links, url)
                                for nl in new_links:
                                    if nl not in visited and nl not in queue:
                                        queue.append(nl)
                    
                    except Exception as e:
                        log(f"  Error: {link[:40]}: {e}")
                
                # === Ensure minimum pages ===
                if ctx.pages_crawled < self.min_pages and queue:
                    log(f"Phase 3: Need {self.min_pages - ctx.pages_crawled} more pages...")
                    
                    for link in queue:
                        if ctx.pages_crawled >= self.min_pages:
                            break
                        if link in visited:
                            continue
                        
                        visited.add(link)
                        try:
                            result = await crawler.arun(link, config=crawler_config)
                            if result.success:
                                ctx.pages_crawled += 1
                                ptype = classify_page_type(link)
                                ctx.page_types_found[ptype] = ctx.page_types_found.get(ptype, 0) + 1
                                log(f"  [{ctx.pages_crawled}] {ptype}: {link[:50]}...")
                        except:
                            pass
        
        # Trim lists
        ctx.sample_products = ctx.sample_products[:25]
        ctx.categories = ctx.categories[:15]
        
        log(f"Crawl complete. {ctx.pages_crawled} pages analyzed.")
        return ctx


# =============================================================================
# Convenience Functions
# =============================================================================

def crawl_site(url: str, max_pages: int = 15, headless: bool = True, progress_callback=None) -> SiteContext:
    """Crawl a site using Crawl4AI."""
    crawler = Crawl4AICrawler(max_pages=max_pages, min_pages=5, headless=headless)
    return crawler.crawl(url, progress_callback)


# Compatibility alias
PLAYWRIGHT_OK = CRAWL4AI_OK
