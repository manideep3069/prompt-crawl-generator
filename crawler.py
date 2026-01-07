"""
Web Crawler for Streamlit Cloud (No Browser Required)

Uses requests + BeautifulSoup for cloud-compatible crawling.
No Playwright or headless browsers needed.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import tldextract

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False

# Cloud-compatible - always true
CRAWL4AI_OK = True


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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

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

def fetch_page(url: str, timeout: int = 15) -> tuple[str, int]:
    """Fetch a page and return (html, status_code)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        return resp.text, resp.status_code
    except Exception:
        return "", 0

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


def extract_nav_sections(soup: BeautifulSoup) -> List[str]:
    sections = []
    
    for nav in soup.select("nav, header, [role='navigation']"):
        for link in nav.select("a"):
            text = clean_text(link.get_text())
            if 2 < len(text) < 25 and text.lower() not in STOP_NAV_WORDS:
                if text not in sections:
                    sections.append(text)
    
    return sections[:20]


def extract_categories(soup: BeautifulSoup) -> tuple[List[str], Dict[str, List[str]]]:
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


def extract_filters(soup: BeautifulSoup) -> tuple[List[str], Dict[str, List[str]]]:
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


def extract_products(soup: BeautifulSoup) -> List[str]:
    products = []
    
    for product in soup.select("[class*='product'], [class*='item'], article, [data-testid*='product']"):
        for name_el in product.select("h2, h3, h4, [class*='name'], [class*='title']"):
            name = clean_text(name_el.get_text())
            if name and 5 < len(name) < 80 and name not in products:
                products.append(name)
                break
    
    return products[:25]


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract internal links."""
    links = []
    
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if href and not href.startswith(("#", "javascript:", "mailto:")):
            full_url = urljoin(base_url, href)
            if same_site(base_url, full_url):
                if not re.search(r'\.(jpg|png|gif|css|js|pdf)(\?|$)', full_url, re.I):
                    if full_url not in links:
                        links.append(full_url)
    
    return links[:50]


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
# Cloud-Compatible Crawler
# =============================================================================

class Crawl4AICrawler:
    """
    Cloud-compatible crawler using requests + BeautifulSoup.
    Works on Streamlit Cloud without browser installation.
    """
    
    def __init__(
        self,
        max_pages: int = 15,
        min_pages: int = 5,
        headless: bool = True,  # Ignored, for compatibility
        timeout_ms: int = 15000
    ):
        self.max_pages = max(max_pages, min_pages)
        self.min_pages = min_pages
        self.timeout = timeout_ms // 1000
    
    def crawl(self, url: str, progress_callback=None) -> SiteContext:
        url = ensure_http(url)
        domain = get_domain(url)
        
        ctx = SiteContext(url=url, domain=domain, page_types_found=defaultdict(int))
        
        def log(msg: str):
            ctx.crawl_notes.append(msg)
            if progress_callback:
                progress_callback(msg)
        
        log(f"Starting crawl of {url}")
        log(f"Target: min {self.min_pages} pages, max {self.max_pages} pages")
        
        visited: Set[str] = set()
        to_visit: List[str] = []
        
        # === Crawl homepage ===
        log("Phase 1: Fetching homepage...")
        
        html, status = fetch_page(url, self.timeout)
        
        if status == 200 and html:
            ctx.pages_crawled += 1
            ctx.page_types_found["homepage"] = 1
            visited.add(url)
            
            soup = BeautifulSoup(html, "lxml")
            text = clean_text(soup.get_text(" "))
            
            # Title and description
            ctx.title = soup.title.get_text() if soup.title else ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            ctx.description = meta_desc.get("content", "")[:300] if meta_desc else ""
            
            # Site type and vocabulary
            ctx.site_type = detect_site_type(url, text)
            log(f"  Site type: {ctx.site_type}")
            
            vocab = detect_vocabulary(text)
            ctx.currency = vocab["currency"]
            ctx.cart_word = vocab["cart_word"]
            ctx.add_phrase = vocab["add_phrase"]
            ctx.signin_word = vocab["signin_word"]
            
            # Features
            features = detect_features(text)
            ctx.has_search = features["has_search"]
            ctx.has_checkout = features["has_checkout"]
            ctx.has_account = features["has_account"]
            ctx.has_wishlist = features["has_wishlist"]
            ctx.guest_checkout = features["guest_checkout"]
            
            # Navigation
            ctx.main_sections = extract_nav_sections(soup)
            log(f"  Found {len(ctx.main_sections)} nav sections")
            
            # Categories
            cats, subcats = extract_categories(soup)
            ctx.categories = cats
            ctx.subcategories = subcats
            
            # Products
            ctx.sample_products = extract_products(soup)
            
            # Links for Phase 2
            to_visit = extract_links(soup, url)
            log(f"  Found {len(to_visit)} internal links")
        else:
            log(f"  Homepage fetch failed (status: {status})")
        
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
            
            queue = priority[:15] + normal[:10]
            
            for link in queue:
                if ctx.pages_crawled >= self.max_pages:
                    break
                if link in visited:
                    continue
                
                visited.add(link)
                
                html, status = fetch_page(link, self.timeout)
                
                if status == 200 and html:
                    ctx.pages_crawled += 1
                    ptype = classify_page_type(link)
                    ctx.page_types_found[ptype] = ctx.page_types_found.get(ptype, 0) + 1
                    
                    log(f"  [{ctx.pages_crawled}] {ptype}: {link[:50]}...")
                    
                    soup = BeautifulSoup(html, "lxml")
                    
                    # Extract more content
                    if ptype in ["category", "search"]:
                        ftypes, fvals = extract_filters(soup)
                        for ft in ftypes:
                            if ft not in ctx.filter_types:
                                ctx.filter_types.append(ft)
                        for k, v in fvals.items():
                            if k not in ctx.filter_values:
                                ctx.filter_values[k] = []
                            ctx.filter_values[k].extend([x for x in v if x not in ctx.filter_values[k]])
                    
                    # More products
                    products = extract_products(soup)
                    for p in products:
                        if p not in ctx.sample_products and len(ctx.sample_products) < 30:
                            ctx.sample_products.append(p)
                    
                    # More categories
                    more_cats, more_subs = extract_categories(soup)
                    for c in more_cats:
                        if c not in ctx.categories:
                            ctx.categories.append(c)
                    ctx.subcategories.update(more_subs)
                    
                    # More links
                    new_links = extract_links(soup, url)
                    for nl in new_links:
                        if nl not in visited and nl not in queue:
                            queue.append(nl)
                
                time.sleep(0.3)  # Be polite
            
            # === Ensure minimum pages ===
            if ctx.pages_crawled < self.min_pages and queue:
                log(f"Phase 3: Need {self.min_pages - ctx.pages_crawled} more pages...")
                
                for link in queue:
                    if ctx.pages_crawled >= self.min_pages:
                        break
                    if link in visited:
                        continue
                    
                    visited.add(link)
                    html, status = fetch_page(link, self.timeout)
                    
                    if status == 200 and html:
                        ctx.pages_crawled += 1
                        ptype = classify_page_type(link)
                        ctx.page_types_found[ptype] = ctx.page_types_found.get(ptype, 0) + 1
                        log(f"  [{ctx.pages_crawled}] {ptype}: {link[:50]}...")
                    
                    time.sleep(0.3)
        
        # Trim lists
        ctx.sample_products = ctx.sample_products[:25]
        ctx.categories = ctx.categories[:15]
        
        log(f"Crawl complete. {ctx.pages_crawled} pages analyzed.")
        return ctx


# =============================================================================
# Convenience Functions
# =============================================================================

def crawl_site(url: str, max_pages: int = 15, headless: bool = True, progress_callback=None) -> SiteContext:
    """Crawl a site using requests (cloud-compatible)."""
    crawler = Crawl4AICrawler(max_pages=max_pages, min_pages=5)
    return crawler.crawl(url, progress_callback)
