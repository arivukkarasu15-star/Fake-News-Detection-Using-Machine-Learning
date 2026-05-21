# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import string
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from functools import lru_cache
import time

# Simple TTL cache for RSS feeds — avoids refetching same feeds on every request
_rss_cache = {}
_RSS_TTL = 300  # 5 minutes

def _cached_fetch_rss(url, source_name, is_tamil=False):
    """Wrapper around fetch_rss_articles with 5-minute TTL caching."""
    now = time.time()
    if url in _rss_cache:
        articles, ts = _rss_cache[url]
        if now - ts < _RSS_TTL:
            return articles
    articles = fetch_rss_articles(url, source_name, is_tamil)
    _rss_cache[url] = (articles, now)
    return articles

# Constants for AI Verification
# TODAY_DATE is dynamically generated within functions to avoid stale dates
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
from deep_translator import GoogleTranslator
from google import genai
from groq import Groq
import json

# Load persistent API keys from .env
load_dotenv()

app = Flask(__name__)

# --- Model Loading Logic ---
print("Loading model and vectorizer...")

def load_model():
    try:
        if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
            print("Model files not found. Creating them now...")
            # Fallback: Run training script if files don't exist
            import train_and_save_model
            train_and_save_model.train_and_save_model()
            
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        print("Model loaded successfully!")
        return vectorizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

vectorizer, model = load_model()

# --- Routes ---

def check_tamil_rss_sources(query):
    """Checks if the query matches recently fetched headlines from top Tamil RSS feeds."""
    if not query:
        return False, None
        
    try:
        # If the query is mostly English, translate it to Tamil for matching
        tamil_query = query
        if re.search('[a-zA-Z]', query) and not re.search('[\u0B80-\u0BFF]', query):
            try:
                tamil_query = GoogleTranslator(source='auto', target='ta').translate(query)
            except Exception as e:
                print(f"Translation error in Tamil RSS check: {e}")

        query_words = set(re.sub('[^a-zA-Z\u0B80-\u0BFF]', ' ', tamil_query).lower().split())
        if not query_words:
            return False, None

        # Fetch articles from multiple Tamil sources
        articles = get_top_tamil_articles()
        
        for article in articles:
            title = article.get('title')
            if not title: continue
            
            title_words = set(re.sub('[^a-zA-Z\u0B80-\u0BFF]', ' ', title).lower().split())
            overlap = query_words.intersection(title_words)
            
            ratio = len(overlap) / len(query_words) if len(query_words) > 0 else 0
            # Keyword overlap check using ratio to prevent loose matches
            if ratio >= 0.50 and len(overlap) >= 2:
                match = {
                    'title': title,
                    'url': article.get('url'),
                    'source': {'name': f"{(article.get('source') or {}).get('name', 'Tamil News')} (Verified Source)"}
                }
                return True, match
    except Exception as e:
        print(f"Tamil RSS Check error: {e}")
    return False, None

def fetch_rss_articles(url, source_name, is_tamil=False):
    """Generic helper to fetch and parse articles from an RSS feed."""
    articles = []
    try:
        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.content)
        for item in root.findall('.//item'):
            # BUG-A FIX: item.find('title') can return None when the tag is absent,
            # causing .text to raise AttributeError and killing the entire feed.
            title = getattr(item.find('title'), 'text', None)
            if not title: continue
            
            # Google News RSS titles often contain the source name at the end, e.g. "Title - Polimer News"
            # We can optionally clean this up for the UI
            clean_title = re.sub(f" - {source_name}$", "", title)

            # BUG-B FIX: same safe access for <link> tag
            article_url = getattr(item.find('link'), 'text', None)

            articles.append({
                'title': clean_title,
                'source': {'name': source_name},
                'url': article_url,
                'urlToImage': None,
                'is_tamil': is_tamil
            })
    except Exception as e:
        print(f"Error fetching RSS from {source_name} ({url}): {e}")
    return articles

def get_top_english_articles(api_key, query=None):
    """Fetches articles from curated high-authority English sources and dynamic search."""
    articles = []
    
    # 0. Dynamic Google News Search for the specific query
    if query:
        # Use top keywords to form a solid search
        from urllib.parse import quote
        q_clean = quote(query[:100])
        dynamic_news = fetch_rss_articles(f"https://news.google.com/rss/search?q={q_clean}&hl=en-US&gl=US&ceid=US:en", "Google News Search")
        articles.extend(dynamic_news)

    # 1. RSS Feeds (Always Available)
    # ── Global wire services & broadcasters ──
    reuters_world  = _cached_fetch_rss("https://feeds.reuters.com/reuters/worldNews",          "Reuters")
    reuters_top    = _cached_fetch_rss("https://feeds.reuters.com/reuters/topNews",            "Reuters Top")
    ap_top         = _cached_fetch_rss("https://feeds.apnews.com/rss/apf-topnews",            "AP News")
    al_jazeera     = _cached_fetch_rss("https://www.aljazeera.com/xml/rss/all.xml",           "Al Jazeera")
    guardian_world = _cached_fetch_rss("https://www.theguardian.com/world/rss",               "The Guardian")
    npr_news       = _cached_fetch_rss("https://feeds.npr.org/1001/rss.xml",                  "NPR News")
    wapo_world     = _cached_fetch_rss("https://feeds.washingtonpost.com/rss/world",          "Washington Post")
    bbc_world      = _cached_fetch_rss("https://feeds.bbci.co.uk/news/world/rss.xml",         "BBC News")
    bbc_uk         = _cached_fetch_rss("https://feeds.bbci.co.uk/news/uk/rss.xml",            "BBC UK")

    # ── CNN ──
    cnn_top      = _cached_fetch_rss("http://rss.cnn.com/rss/cnn_topstories.rss",            "CNN")
    cnn_world    = _cached_fetch_rss("http://rss.cnn.com/rss/cnn_world.rss",                 "CNN World")
    cnn_us       = _cached_fetch_rss("http://rss.cnn.com/rss/cnn_us.rss",                    "CNN US")
    cnn_politics = _cached_fetch_rss("http://rss.cnn.com/rss/cnn_allpolitics.rss",           "CNN Politics")
    cnn_business = _cached_fetch_rss("http://rss.cnn.com/rss/money_news_international.rss",  "CNN Business")

    # ── India English ──
    indian_express  = _cached_fetch_rss("https://news.google.com/rss/search?q=The+Indian+Express&hl=en-IN&gl=IN&ceid=IN:en", "The Indian Express")
    the_hindu       = _cached_fetch_rss("https://news.google.com/rss/search?q=The+Hindu&hl=en-IN&gl=IN&ceid=IN:en",          "The Hindu")
    times_of_india  = _cached_fetch_rss("https://news.google.com/rss/search?q=Times+of+India&hl=en-IN&gl=IN&ceid=IN:en",    "Times of India")
    ndtv_top        = _cached_fetch_rss("https://feeds.feedburner.com/ndtvnews-top-stories",  "NDTV")
    hindustan_times = _cached_fetch_rss("https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml", "Hindustan Times")
    india_today     = _cached_fetch_rss("https://www.indiatoday.in/rss/1206514",              "India Today")
    economic_times  = _cached_fetch_rss("https://economictimes.indiatimes.com/rssfeedsdefault.cms", "Economic Times")
    deccan_herald   = _cached_fetch_rss("https://www.deccanherald.com/rss-feed/section-4",   "Deccan Herald")
    news18_india    = _cached_fetch_rss("https://www.news18.com/rss/india.xml",              "News18")
    the_wire        = _cached_fetch_rss("https://thewire.in/feed",                           "The Wire")
    scroll_in       = _cached_fetch_rss("https://scroll.in/feed",                            "Scroll.in")

    # Merge — wire services & CNN first (highest corroboration priority)
    all_rss_sources = [
        reuters_world, reuters_top, ap_top,
        cnn_top, cnn_world, cnn_us, cnn_politics, cnn_business,
        bbc_world, bbc_uk, al_jazeera, guardian_world, npr_news, wapo_world,
        indian_express, the_hindu, times_of_india, ndtv_top,
        hindustan_times, india_today, economic_times, deccan_herald,
        news18_india, the_wire, scroll_in,
    ]
    for i in range(max((len(s) for s in all_rss_sources), default=0)):
        for source in all_rss_sources:
            if i < len(source):
                articles.append(source[i])

    # 2. NewsAPI (If available)
    if api_key:
        sources = "reuters,associated-press,bbc-news,cnn,al-jazeera-english,the-guardian-uk,the-washington-post,the-new-york-times,the-wall-street-journal,ndtv,the-times-of-india,the-hindu,hindustan-times,india-today"
        url = f"https://newsapi.org/v2/top-headlines?sources={sources}&pageSize=80&apiKey={api_key}"
        try:
            news_data = requests.get(url, timeout=10).json()
            if news_data.get("status") == "ok":
                articles.extend(news_data.get("articles", []))
        except Exception as e:
            print(f"Error fetching top English articles from NewsAPI: {e}")
            
    return articles

def check_english_rss_sources(query, api_key):
    """Checks if the query matches recently fetched headlines using strict ratio matching."""
    if not query:
        return False, None
        
    try:
        query_roots, query_words = get_clean_words(query)
        q_count = len(query_roots)
        if not q_count:
            return False, None

        # Geographical and critical words
        geo_words = {'indonesia', 'sumatra', 'java', 'tamil', 'nadu', 'chennai', 'india', 'bali', 'jakarta', 'delhi', 'mumbai'}
        query_geo = query_roots.intersection(geo_words)
        query_nums = {w for w in query_roots if any(c.isdigit() for c in w)}

        # Fetch articles from multiple English sources
        articles = get_top_english_articles(api_key, query=query)

        # FIX: Also search Google News RSS with the raw query so economic/political
        # headlines like "US gas hits $4..." are found even if $ is stripped by get_clean_words.
        from urllib.parse import quote as _qenc
        raw_rss = fetch_rss_articles(
            f"https://news.google.com/rss/search?q={_qenc(query[:100])}&hl=en-US&gl=US&ceid=US:en",
            "Google News Direct"
        )
        articles = raw_rss + articles  # raw-query results take priority

        for article in articles:
            title = article.get('title')
            if not title: continue
            
            art_roots, art_words_orig = get_clean_words(title)
            matching_words = query_roots.intersection(art_roots)
            m_count = len(matching_words)
            
            # Bonus weighting
            if query_geo.intersection(art_roots): m_count += 1
            if query_nums.intersection(art_roots): m_count += 0.5
            
            ratio = m_count / min(7, q_count) if q_count > 0 else 0
            
            if ratio >= 0.60:
                match = {
                    'title': title,
                    'url': article.get('url'),
                    'source': {'name': f"{(article.get('source') or {}).get('name', 'English News')} (Verified Source)"},
                    'match_score': ratio * 100
                }
                return True, match
                
    except Exception as e:
        print(f"English RSS Check error: {e}")
    return False, None

# Helper for consistent tokenization and stemming across verification functions
def get_clean_words(text):
    if not text: return set(), set()  # BUG-1 FIX: always return 2-tuple; single set() caused ValueError on unpack
    # Expanded stop words for better query extraction
    stop_words = {
        'is', 'a', 'the', 'in', 'on', 'at', 'by', 'of', 'for', 'with', 'and', 'or', 'but', 'are', 'was', 'were', 'to', 'has', 'have', 'had', 'that', 'this',
        'it', 'an', 'as', 'if', 'from', 'about', 'who', 'what', 'when', 'where', 'how', 'which', 'their', 'his', 'her', 'its', 'they', 'them'
    }
    # Keep alphanumeric characters and space
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text_clean.split()
    
    # Returns both filtered original words and stemmed roots
    words_orig = [w for w in words if w not in stop_words and len(w) > 2]
    
    roots = []
    for w in words_orig:
        if w.isascii():
            root = w
            if w.endswith('ian'): root = w[:-3]
            elif w.endswith('s') and not w.endswith('ss'): root = w[:-1]
            elif w.endswith('ed'): root = w[:-2]
            elif w.endswith('ing'): root = w[:-3]
            roots.append(root)
        else:
            roots.append(w)
            
    return set(roots), set(words_orig)

def get_top_tamil_articles():
    """Fetches articles from curated high-authority Tamil sources."""
    try:
        # ── Direct RSS feeds (most reliable) ──
        bbc_tamil    = _cached_fetch_rss("https://feeds.bbci.co.uk/tamil/rss.xml",         "BBC Tamil",          is_tamil=True)
        bbc_tamil_ix = _cached_fetch_rss("https://www.bbc.com/tamil/index.xml",            "BBC Tamil",          is_tamil=True)
        dinamalar    = _cached_fetch_rss("https://www.dinamalar.com/rss_feed.asp",         "Dinamalar",          is_tamil=True)
        dinamani     = _cached_fetch_rss("https://www.dinamani.com/rss/",                  "Dinamani",           is_tamil=True)
        vikatan      = _cached_fetch_rss("https://www.vikatan.com/rss.xml",                "Vikatan",            is_tamil=True)
        maalai_malar = _cached_fetch_rss("https://www.maalaimalar.com/feed",               "Maalai Malar",       is_tamil=True)
        oneindia_ta  = _cached_fetch_rss("https://tamil.oneindia.com/rss/tamil-news-fb.xml","OneIndia Tamil",    is_tamil=True)
        tamil_murasu = _cached_fetch_rss("https://www.tamilmurasu.com.sg/feed",            "Tamil Murasu",       is_tamil=True)

        # ── Google News Tamil search feeds ──
        polimer      = _cached_fetch_rss("https://news.google.com/rss/search?q=Polimer+News&hl=ta&gl=IN&ceid=IN:ta",         "Polimer News",       is_tamil=True)
        news7        = _cached_fetch_rss("https://news.google.com/rss/search?q=News7+Tamil&hl=ta&gl=IN&ceid=IN:ta",          "News 7 Tamil",       is_tamil=True)
        thanthi      = _cached_fetch_rss("https://news.google.com/rss/search?q=Daily+Thanthi&hl=ta&gl=IN&ceid=IN:ta",        "Daily Thanthi",      is_tamil=True)
        puthiya      = _cached_fetch_rss("https://news.google.com/rss/search?q=Puthiya+Thalamurai&hl=ta&gl=IN&ceid=IN:ta",  "Puthiya Thalamurai", is_tamil=True)
        thanthi_tv   = _cached_fetch_rss("https://news.google.com/rss/search?q=Thanthi+TV&hl=ta&gl=IN&ceid=IN:ta",          "Thanthi TV",         is_tamil=True)
        ndtv_tamil   = _cached_fetch_rss("https://news.google.com/rss/search?q=NDTV+Tamil&hl=ta&gl=IN&ceid=IN:ta",          "NDTV Tamil",         is_tamil=True)
        sun_news     = _cached_fetch_rss("https://news.google.com/rss/search?q=Sun+News+Tamil&hl=ta&gl=IN&ceid=IN:ta",      "Sun News",           is_tamil=True)
        kalaignar    = _cached_fetch_rss("https://news.google.com/rss/search?q=Kalaignar+TV+Tamil&hl=ta&gl=IN&ceid=IN:ta",  "Kalaignar TV",       is_tamil=True)
        nakkheeran   = _cached_fetch_rss("https://news.google.com/rss/search?q=Nakkheeran+Tamil&hl=ta&gl=IN&ceid=IN:ta",    "Nakkheeran",         is_tamil=True)

        # Direct feeds first, then Google News feeds
        all_sources = [
            bbc_tamil, bbc_tamil_ix, dinamalar, dinamani, vikatan,
            maalai_malar, oneindia_ta, tamil_murasu,
            polimer, news7, thanthi, puthiya, thanthi_tv,
            ndtv_tamil, sun_news, kalaignar, nakkheeran,
        ]
        articles = []
        max_len = max(len(s) for s in all_sources) if all_sources else 0
        for i in range(max_len):
            for source in all_sources:
                if i < len(source):
                    articles.append(source[i])
        return articles
    except Exception as e:
        print(f"Error fetching top Tamil articles: {e}")
        return []

def check_fact_checking_apis(query, full_text, newsapi_key, serpapi_key):
    api_error = False
    
    # 1. CLEANING & TOKENIZATION
    # (Uses global get_clean_words)

    query_roots, query_words_orig = get_clean_words(query)
    q_count = len(query_roots)
    if not q_count:
        return False, False, None, "Unverified", False
    
    # Core geographical targets
    geo_words = {'indonesia', 'sumatra', 'java', 'tamil', 'nadu', 'chennai', 'india', 'bali', 'jakarta'}
    query_geo = query_roots.intersection(geo_words)

    # 2. TRUSTED NEWS CORROBORATION — Stage 1: Top Headlines
    if newsapi_key:
        try:
            # Match ORIGINAL words for search, but sorted by importance
            core_words = sorted([w for w in query_words_orig if w not in geo_words], key=len, reverse=True)[:2]
            search_str = " ".join(list(query_geo)[:2] + core_words)
            
            for use_sources in [False, True]:
                src_param = "&sources=reuters,bbc-news,the-new-york-times,the-wall-street-journal,the-times-of-india,the-hindu" if use_sources else ""
                fc_url = f"https://newsapi.org/v2/top-headlines?q={requests.utils.quote(search_str)}&apiKey={newsapi_key}&pageSize=20{src_param}"
                fc_res = requests.get(fc_url, timeout=10).json()
                
                if fc_res.get('status') == 'ok':
                    for art in fc_res.get('articles', []):
                        art_title = art.get('title', '')
                        if not art_title: continue
                        art_roots, _ = get_clean_words(art_title)
                        matching_words = query_roots.intersection(art_roots)
                        m_count = len(matching_words)
                        
                        # Bonus: If geographical targets match, significantly boost score
                        if query_geo.intersection(art_roots):
                            m_count += 1
                            
                        effective_q = min(7, q_count)
                        match_ratio = m_count / effective_q
                        if match_ratio >= 0.45:
                            match_percentage = min(100, match_ratio * 100)
                            # BUG-E FIX: art.get('source', {}) returns None (not {}) when
                            # source is explicitly null in the NewsAPI JSON response.
                            source_name_fc = (art.get('source') or {}).get('name', 'Trusted News')
                            return False, False, {**art, 'match_score': match_percentage, 'matched_words': list(matching_words)}, f"Corroborated by {source_name_fc} ({match_percentage:.0f}% match)", False
                elif fc_res.get('code') in ['apiKeyInvalid', 'apiKeyDisabled']:
                    api_error = True
                    break
        except Exception as e:
            print(f"Trusted source matching error: {e}")

    # 3. FACT-CHECK DOMAINS
    if serpapi_key:
        try:
            serp_url = f"https://serpapi.com/search.json?q={requests.utils.quote(query[:150])}&api_key={serpapi_key}&num=5"
            res = requests.get(serp_url, timeout=10).json()
            if "error" in res and ("Invalid API key" in res["error"] or "unauthorized" in res["error"].lower()):
                 api_error = True
            if "organic_results" in res:
                for result in res["organic_results"]:
                    link = result.get("link", "").lower()
                    if any(domain in link for domain in ['snopes.com', 'politifact.com', 'factcheck.org', 'reuters.com/fact-check', 'pib.gov.in']):
                        snippet = result.get("snippet", "").lower()
                        title = result.get("title", "").lower()
                        art_roots, _ = get_clean_words(title + " " + snippet)
                        if len(query_roots.intersection(art_roots)) / q_count >= 0.35:
                            if any(word in (snippet + title) for word in ['false', 'fake', 'hoax', 'debunked', 'unproven', 'misleading']):
                                return True, False, {'title': result.get("title", ""), 'url': result['link'], 'source': {'name': 'Fact Checker'}}, "Debunked by Fact Checkers", False
                            if any(word in (snippet + title) for word in ['true', 'accurate', 'correct', 'mostly true', 'verified']):
                                return False, True, {'title': result.get("title", ""), 'url': result['link'], 'source': {'name': 'Fact Checker'}}, "Verified by Fact Checkers", False
        except Exception as e:
            print(f"SerpApi error: {e}")

    # 4. BROAD NEWS — Multi-stage "everything" search
    if newsapi_key and not api_error:
        try:
            # Build smarter search trials using ORIGINAL words
            alpha_words = sorted([w for w in query_words_orig if not w.isdigit()], key=len, reverse=True)
            # Remove geo words from alpha words to avoid duplicates in search string
            search_keywords = [w for w in alpha_words if w not in geo_words]
            
            # Focused search trials
            search_trials = []
            if query_geo:
                # Trial 0: Geo + Top 2 Keywords (e.g., 'indonesia floods landslides')
                search_trials.append(" ".join(list(query_geo)[:2] + search_keywords[:2]))
                # Trial 1: Geo + 'deaths' or 'displacement' if present
                priority = [w for w in search_keywords if w in {'deaths', 'dead', 'million', 'displac'}]
                if priority:
                    search_trials.append(" ".join(list(query_geo)[:1] + priority[:2]))
            
            # Trial 2: Top 4 keywords without geo
            search_trials.append(" ".join(search_keywords[:4]))
            
            # Trial 3: Raw snippet (fallback)
            search_trials.append(re.sub(r'[^\w\s]', ' ', query)[:100].strip())
            
            seen = set()
            search_trials = [t for t in search_trials if t and not (t in seen or seen.add(t))]

            for trial_idx, q_term in enumerate(search_trials):
                # Loose thresholds (0.35 - 0.5) to capture varying headlines
                match_threshold = 0.45 if trial_idx == 0 else (0.40 if trial_idx == 1 else 0.35)
                en_url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(q_term)}&language=en&pageSize=15&sortBy=relevancy&apiKey={newsapi_key}"
                en_res = requests.get(en_url, timeout=10).json()
                
                if en_res.get('status') == 'ok' and en_res.get('totalResults', 0) > 0:
                    for art in en_res['articles'][:10]:
                        art_title = art.get('title', '')
                        art_roots, _ = get_clean_words(art_title)
                        matching_words = query_roots.intersection(art_roots)
                        m_count = len(matching_words)
                        
                        # Weighting bonus
                        disaster_keywords = {'flood', 'landslid', 'death', 'kill', 'displac', 'disaster', 'shelter'}
                        if query_geo.intersection(art_roots):
                            m_count += 1.5 # Increased
                        # Check sub-region matches (e.g. Sumatra in Indonesia)
                        if ('sumatra' in art_roots or 'java' in art_roots) and 'indonesia' in query_roots:
                            m_count += 1.0
                            
                        # Disaster keyword overlap bonus
                        if any(w in art_roots for w in disaster_keywords) and any(w in query_roots for w in disaster_keywords):
                            m_count += 1.0

                        ratio = m_count / min(7, q_count)

                        if ratio >= match_threshold:
                            match_percentage = min(100, ratio * 100)
                            # BUG-E FIX: (art.get('source') or {}) handles explicitly-null source
                            source_name = (art.get('source') or {}).get('name', 'News Source')
                            return False, False, {**art, 'match_score': match_percentage, 'matched_words': list(matching_words)}, f"Corroborated by {source_name} ({match_percentage:.0f}% match)", False
                elif en_res.get('code') in ['apiKeyInvalid', 'apiKeyMissing']:
                    api_error = True
                    break
        except Exception as e:
            print(f"Global NewsAPI error: {e}")

    return False, False, None, "Unverified", api_error

def check_semantic_consistency(claim, matched_title, groq_key, gemini_key=None):
    """
    Uses Groq (Llama 3) or Gemini as fallback to perform a semantic check to ensure
    that the matched headline actually supports the user's claim, preventing
    false keyword matches (e.g. "India lost" matching "India won").
    Returns True if the claim is supported by the title, False otherwise.
    """
    if not claim or not matched_title:
        return True
    
    prompt_text = f"""
    User Claim: "{claim}"
    News Headline: "{matched_title}"
    
    Does the News Headline SPECIFICALLY confirm or support the User Claim?
    
    Rules:
    - Answer 'NO' if the headline contradicts the claim (e.g. claim says lost, headline says won).
    - Answer 'NO' if the headline is about a RELATED but DIFFERENT topic (e.g. claim is about winning a tournament, but headline is a player profile or a different match).
    - Answer 'NO' if the headline does NOT confirm the SPECIFIC event or outcome stated in the claim.
    - Answer 'YES' ONLY if the headline directly confirms the exact claim being made.
    
    Respond ONLY with the exact word 'YES' or 'NO'. No other text.
    """
    
    # Try Groq first (fastest)
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a strict semantic logical analyzer."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.0,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip().upper()
            return 'NO' not in answer
        except Exception as e:
            print(f"Semantic Consistency Check (Groq) error: {e}")
    
    # Fallback to Gemini
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt_text
            )
            if response.text:
                answer = response.text.strip().upper()
                return 'NO' not in answer
        except Exception as e:
            print(f"Semantic Consistency Check (Gemini) error: {e}")
    
    # Default to True only if NEITHER key is available
    return True

def get_live_context(query, serpapi_key, newsapi_key):
    """Fetches real-time search snippets from SerpApi, NewsAPI, or free Google News RSS to give context to Groq/OpenRouter.
    Returns (context_str, source_urls) where source_urls is a list of article URLs for attribution."""
    context_str = ""
    source_urls = []
    # Try SerpApi first for quick web snippets
    if serpapi_key:
        try:
            serp_url = f"https://serpapi.com/search.json?q={requests.utils.quote(query[:150])}&api_key={serpapi_key}&num=3"
            res = requests.get(serp_url, timeout=5).json()
            if "organic_results" in res:
                snippets = []
                for result in res["organic_results"][:3]:
                    snippets.append(f"- {result.get('title')}: {result.get('snippet')}")
                if snippets:
                    context_str += "Recent Search Results:\n" + "\n".join(snippets) + "\n\n"
                    source_urls += [r.get("link") for r in res["organic_results"][:3] if r.get("link")]
        except Exception as e:
            print(f"SerpApi context fetch error: {e}")
            
    # Try NewsAPI if no serpapi or as backup
    if newsapi_key and len(context_str) < 50:
        try:
            en_url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query[:100])}&language=en&pageSize=3&sortBy=relevancy&apiKey={newsapi_key}"
            en_res = requests.get(en_url, timeout=5).json()
            if en_res.get('status') == 'ok':
                snippets = []
                for art in en_res.get('articles', [])[:3]:
                    snippets.append(f"- {art.get('title')}: {art.get('description')}")
                if snippets:
                    context_str += "Recent News Articles:\n" + "\n".join(snippets) + "\n\n"
                    source_urls += [a.get("url") for a in en_res.get("articles", [])[:3] if a.get("url")]
        except Exception as e:
             print(f"NewsAPI context fetch error: {e}")

    # --- FREE FALLBACK: Google News RSS (no API key needed) ---
    # Always runs if context is still empty, ensuring Groq gets real-time headlines
    if len(context_str) < 50:
        try:
            from urllib.parse import quote as url_quote
            import datetime as _dt
            # Preserve outcome words (won/lost/beat etc.) so RSS targets the specific result
            outcome_words = {'won', 'win', 'wins', 'lost', 'lose', 'beat', 'beats',
                             'defeated', 'champion', 'title', 'result', 'final', 'score'}
            # FIX: Extract currency/price tokens BEFORE get_clean_words strips $ symbols
            # e.g. "$4" becomes "4dollar" so Groq context search finds economic news
            import re as _re
            currency_tokens = _re.findall(r'\$[\d,.]+', query)
            currency_str = ' '.join(t.replace('$', '') + 'dollar' for t in currency_tokens)

            query_roots, query_words_orig = get_clean_words(query)
            stop = {'said', 'say', 'says', 'new', 'year', 'day', 'time', 'back'}
            priority = [w for w in query_words_orig if w in outcome_words]
            rest = [w for w in query_roots if len(w) > 3 and w not in stop and w not in set(priority)]
            keywords = (priority + rest)[:6]

            # FIX: If too few keywords extracted (non-sports headlines), use cleaned raw query.
            # This prevents vague RSS searches that cause Groq to say FAKE for real news.
            if len(keywords) < 3:
                rss_query_str = _re.sub(r"[^\w\s$]", ' ', query).strip()[:100]
            else:
                base = " ".join(keywords)
                rss_query_str = (currency_str + ' ' + base).strip() if currency_str else base
            # Date-filtered search (last 7 days) to avoid stale articles
            week_ago = (_dt.datetime.now() - _dt.timedelta(days=7)).strftime("%Y-%m-%d")
            rss_url = (f"https://news.google.com/rss/search?q={url_quote(rss_query_str)}"
                       f"+after:{week_ago}&hl=en-US&gl=US&ceid=US:en")
            rss_articles = fetch_rss_articles(rss_url, "Google News")
            # Fallback without date filter if filtered search returns nothing
            if not rss_articles:
                rss_url_fb = f"https://news.google.com/rss/search?q={url_quote(rss_query_str)}&hl=en-US&gl=US&ceid=US:en"
                rss_articles = fetch_rss_articles(rss_url_fb, "Google News")
            if rss_articles:
                snippets = [f"- {a['title']}" for a in rss_articles[:6] if a.get('title')]
                if snippets:
                    today = datetime.now().strftime("%B %d, %Y")
                    context_str += f"Live Google News Headlines (as of {today}):\n" + "\n".join(snippets) + "\n\n"
                    print(f"DEBUG: RSS live context fetched ({len(snippets)} headlines) for query: {query[:60]}")
                    source_urls += [a.get("url") for a in rss_articles[:3] if a.get("url")]
        except Exception as e:
            print(f"Google News RSS context fetch error: {e}")
            
    return context_str, source_urls


def get_source_url_for_verdict(claim, context_urls, groq_key, gemini_key=None, verdict=None):
    """
    Returns a source URL based on verdict type:
    - REAL: best-matching context URL, falling back to first available URL, then Google search
    - FAKE (fabricated): returns None — no source needed for completely false claims

    The old logic required 2+ keyword matches between claim words and the URL string.
    This broke for Tamil claims (Tamil words never appear in English news URLs) and for
    claims whose keywords don't appear in URL slugs (e.g. gas price news).
    Fix: try keyword match first, then fall back to the first valid context URL.
    """
    from urllib.parse import quote
    google_search = f"https://www.google.com/search?q={quote(claim[:100])}"

    # Fabricated claims get no source link
    if verdict and str(verdict).upper() == 'FAKE':
        return None

    # Filter out None/empty URLs once
    valid_urls = [u for u in (context_urls or []) if u and u.startswith('http')]
    if not valid_urls:
        return google_search

    # Try keyword match first (works well for ASCII claims with distinctive slug words)
    claim_roots, _ = get_clean_words(claim)
    specific_words = {w for w in claim_roots if len(w) > 5 and w.isascii()}
    if specific_words:
        for url in valid_urls:
            url_lower = url.lower()
            matches = sum(1 for w in specific_words if w in url_lower)
            if matches >= 2:
                return url

    # Keyword match failed (Tamil input, short claim, or keywords not in URL slug).
    # Return the first context URL directly — it is the source Groq/Groq used to
    # verify the claim, so it is the most relevant link available.
    return valid_urls[0]

def check_with_gemini(query, api_key):
    """
    Uses Google Gemini with Google Search Grounding as an intelligent fact-checking fallback.
    Gemini will search the web in real-time before making a verdict.
    Returns (prediction_override, confidence, message, source_name) or None if it fails/opts out.
    """
    if not api_key or not query:
        return None
        
    try:
        from google.genai import types
        client = genai.Client(api_key=api_key)
        
        # Enable Google Search Grounding — Gemini will search the web before answering
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        TODAY_DATE = datetime.now().strftime("%B %d, %Y")
        prompt = f"""
        You are a highly accurate and STRICTLY objective professional fact-checker.
        TODAY'S DATE: {TODAY_DATE}.
        
        CRITICAL INSTRUCTIONS:
        - Use Google Search to find real-time information about the claim below.
        - Search across news, history, geography, politics, science, sports, and any relevant domain.
        - Cross-reference multiple sources before making a verdict.
        - PAY VERY CLOSE ATTENTION TO SPECIFIC DETAILS: dates, years, numbers, names, locations, and statistics.
        - If the general topic is real but a SPECIFIC DETAIL is WRONG (e.g. wrong year, wrong name, wrong number), you MUST mark it as "FAKE".
          Example: "India gained independence in 1950" is FAKE because the correct year is 1947.
          Example: "The capital of Australia is Sydney" is FAKE because the correct capital is Canberra.
        - Only mark as "REAL" if ALL specific details in the claim are factually accurate.
        - If the claim is completely unsupported by any evidence, or contradicts real facts, mark it as "FAKE". You MUST choose REAL or FAKE.
        
        Analyze the following news headline or claim: "{query}"

        After searching, respond ONLY with a raw JSON object (no markdown, no extra text):
        {{
            "status": "REAL" or "FAKE",
            "confidence": an integer between 70 and 99,
            "explanation": "Brief explanation of why this is real or fake, citing what you found. If FAKE, specify which detail is wrong and what the correct fact is.",
            "source_type": "Web Search" or "Historical Record" or "General Knowledge"
        }}
        """
        
        config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=config
        )
        if not response.text:
            return None
            
        text = response.text.strip()
        
        # Remove potential markdown formatting if Gemini disobeys
        if "```" in text:
            text = text.replace("```json", "").replace("```", "")
            
        data = json.loads(text.strip())
        
        # Extract source URLs from grounding metadata if available
        source_url = None
        source_titles = []
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                gm = getattr(candidate, 'grounding_metadata', None)
                if gm:
                    chunks = getattr(gm, 'grounding_chunks', None)
                    if chunks:
                        for chunk in chunks[:3]:
                            web = getattr(chunk, 'web', None)
                            if web:
                                if not source_url:
                                    source_url = getattr(web, 'uri', None)
                                title = getattr(web, 'title', None)
                                if title:
                                    source_titles.append(title)
        except Exception as e:
            print(f"Grounding metadata parse warning: {e}")
        
        status = str(data.get('status', '')).upper()
        # Only override if Gemini is fairly confident
        if data.get('confidence', 0) >= 70 and status in ['REAL', 'FAKE', 'TRUE']:
            final_status = 'real' if status in ['REAL', 'TRUE'] else 'fake'
            explanation = data.get('explanation', 'Verified by Gemini with Google Search.')
            
            # Build source info with grounded references
            source_info = data.get('source_type', 'Web Search')
            if source_titles:
                source_info = f"Web Search via {', '.join(source_titles[:2])}"
            
            result = (
                final_status, 
                float(data['confidence']), 
                f"Google AI (Search-Grounded): {explanation}",
                f"Gemini AI ({source_info})"
            )
            
            # Attach source URL for linking in the UI
            if source_url:
                result = result + (source_url,)
            
            return result
            
    except Exception as e:
        print(f"Gemini AI Check error: {e}")
        
    return None


def check_with_groq(query, api_key, live_context=""):
    """
    Uses Groq (Llama-3-70b) as a secondary highly intelligent fact-checking fallback.
    Returns (prediction_override, confidence, message, source_name) or None if it fails.
    """
    if not api_key or not query:
        return None
        
    try:
        client = Groq(api_key=api_key)
        
        TODAY_DATE = datetime.now().strftime("%B %d, %Y")
        live_info = f"LIVE WEB CONTEXT (use this to inform your verdict):\n{live_context}" if live_context else ""
        
        prompt = f"""
        You are a highly accurate and strictly objective professional fact-checker.
        TODAY\'S DATE: {TODAY_DATE}.
        
        {live_info}
        
        Analyze the following news headline or claim: "{query}"

        Determine if this claim is TRUE (a real event supported by evidence) or FAKE (unsupported, contradicted by facts, or no credible evidence exists).
        
        RULES:
        - PAY CLOSE ATTENTION TO SPECIFIC DETAILS: dates, years, numbers, names, locations, statistics.
        - If a specific detail is WRONG (e.g. wrong year, wrong outcome, wrong location), mark as "FAKE".
        - If LIVE WEB CONTEXT above CONFIRMS the claim with headlines, mark as "REAL".
        - If LIVE WEB CONTEXT shows NO headlines AND the claim is about a local/regional event (local crime, local flood, local election), mark as "FAKE" only if it truly seems implausible.
        - IMPORTANT: If the claim is about a GLOBAL or NATIONAL event (international politics, war, major economy, US/UK/India news, gas prices, stock markets, major sports), and the LIVE WEB CONTEXT is empty or unhelpful, do NOT mark as "FAKE" based on absence alone. Instead set status to "UNVERIFIED" and confidence to 60.
        - Only mark "FAKE" if a headline in the LIVE WEB CONTEXT DIRECTLY AND EXPLICITLY contradicts a specific detail (e.g. a headline says "India WON" but the claim says "India LOST").
        - CRITICAL: The live context is a SMALL SAMPLE of recent headlines. NOT FINDING something in the live context is NOT a contradiction. Absence from the live context does NOT mean the event did not happen.
        - Do NOT say a claim is fake because it is not mentioned in the live context.
        - NEVER mention knowledge cutoff, training data, or lack of real-time access.
        - Write the explanation as a confident factual statement.
        
        Respond ONLY with a raw JSON object, no markdown:
        {{
            "status": "REAL" or "FAKE" or "UNVERIFIED",
            "confidence": an integer between 60 and 99,
            "explanation": "Confident factual explanation. If FAKE, cite the specific wrong detail. If UNVERIFIED, say: Live context was insufficient to confirm or deny this claim.",
            "source_type": "Live News Context" or "Historical Record" or "General Knowledge"
        }}
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional fact-checker that responds only in valid JSON. Never mention knowledge cutoffs or training data. Always give a definitive REAL or FAKE verdict."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=1000,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        text = completion.choices[0].message.content.strip()

        data = json.loads(text)

        status = str(data.get('status', '')).upper()
        explanation = data.get('explanation', '')

        # ABSENCE-OF-EVIDENCE GUARD:
        # If Groq says FAKE but the explanation is just "no sources found" (not a specific
        # factual contradiction), return None so the pipeline stays Unverified rather than
        # actively mislabelling real breaking news as FAKE.
        absence_phrases = [
            'no credible news sources report',
            'no credible sources report',
            'live context does not confirm',
            'live web context does not confirm',
            'no headlines',
            'no news sources',
            'insufficient to confirm',
            'cannot be verified',
            'no evidence',
            'lack of any mention',
            'no mention of',
            'not mentioned',
            'absence of',
            'context instead discusses',
            'context does not include',
            'not found in',
            'not present in',
            'no reference to',
            'no coverage of',
            'not reported',
            'contradicted by the lack',
            'lack of coverage',
            'no reporting',
        ]
        # A real contradiction requires citing what the CORRECT fact is, not just what is missing
        contradiction_phrases = [
            'actually won', 'actually lost', 'actually said', 'actually happened',
            'correct figure is', 'correct year is', 'correct answer is',
            'in reality', 'in fact the', 'the real', 'true figure',
            'headline says', 'source says', 'according to', 'reports that',
        ]
        if status == 'FAKE' and explanation:
            expl_low = explanation.lower()
            is_absence_only = any(p in expl_low for p in absence_phrases)
            has_contradiction = any(p in expl_low for p in contradiction_phrases)
            if is_absence_only and not has_contradiction:
                print(f"DEBUG: Groq absence-of-evidence FAKE suppressed. Explanation: {explanation[:100]}")
                return None  # Let pipeline stay Unverified — do not fabricate a FAKE verdict

        # UNVERIFIED status from new prompt — return None so pipeline stays Unverified
        if status == 'UNVERIFIED':
            print(f"DEBUG: Groq returned UNVERIFIED — passing through to next verifier")
            return None

        # SELF-CONTRADICTION GUARD:
        # Catches cases where Groq says FAKE but its own explanation actually
        # confirms the claim's key numbers/facts. Example: claim says "131 crore",
        # Groq says "BCCI announced 131 crore... not 2131 crore" — the explanation
        # CONFIRMS 131 crore but Groq confused it with a different year's figure.
        if status == 'FAKE' and explanation:
            claim_nums = set(re.findall(r'\d+', query))
            expl_lower = explanation.lower()
            # Phrases that indicate Groq is actually confirming the claim
            confirm_phrases = [
                'announced a cash reward of', 'confirmed that', 'did announce',
                'correctly states', 'is correct', 'has announced', 'did win',
                'did occur', 'is accurate', 'indeed',
            ]
            expl_confirms = any(p in expl_lower for p in confirm_phrases)
            # Key numbers from claim appear in explanation BEFORE any "not X" negation
            expl_before_not = expl_lower.split(' not ')[0] if ' not ' in expl_lower else expl_lower
            nums_confirmed = claim_nums and any(
                num in expl_before_not for num in claim_nums if len(num) >= 2
            )
            if expl_confirms or nums_confirmed:
                print(f"DEBUG: Groq self-contradiction — explanation confirms claim. Skipping FAKE verdict.")
                return None  # Pass to next verifier (Gemini/OpenRouter)

        # Only override if Groq is fairly confident
        if data.get('confidence', 0) >= 70 and status in ['REAL', 'FAKE', 'TRUE']:
            final_status = 'real' if status in ['REAL', 'TRUE'] else 'fake'
            return (
                final_status,
                float(data['confidence']),
                f"Groq AI Verification: {explanation or 'Verified by Groq.'}",
                f"Groq AI ({data.get('source_type', 'Knowledge Base')})"
            )

    except Exception as e:
        print(f"Groq AI Check error: {e}")

    return None

def check_with_openrouter(query, api_key, live_context=""):
    """
    Uses OpenRouter (e.g., Anthropic Claude 3 Haiku or similar) as a tertiary fact-checking fallback.
    Returns (prediction_override, confidence, message, source_name) or None if it fails.
    """
    if not api_key or not query:
        return None
        
    try:
        # Using a reliable and fast model through OpenRouter
        # You can change this to "anthropic/claude-3-haiku" or "google/gemini-2.0-flash-001" etc.
        model_name = "google/gemini-2.0-flash-001" 
        
        TODAY_DATE = datetime.now().strftime("%B %d, %Y")
        live_info = f"LIVE WEB CONTEXT (use this to inform your verdict):\n{live_context}" if live_context else ""
        
        prompt = f"""
        You are a highly accurate and strictly objective professional fact-checker.
        TODAY'S DATE: {TODAY_DATE}.
        
        {live_info}
        
        Analyze the following news headline or claim: "{query}"

        Determine if this claim is TRUE (a real event supported by evidence) or FAKE (unsupported, contradicted by facts, or no credible evidence exists).
        
        RULES:
        - PAY CLOSE ATTENTION TO SPECIFIC DETAILS: dates, years, numbers, names, locations, statistics.
        - If a specific detail is WRONG, mark as "FAKE".
        - If LIVE WEB CONTEXT CONFIRMS the claim, mark as "REAL".
        - If LIVE WEB CONTEXT shows NO headlines AND the claim is about a local/regional event, mark as "FAKE" only if it seems implausible. For GLOBAL/NATIONAL events (war, politics, economy, gas prices), set status "UNVERIFIED" if context is missing — do NOT mark FAKE based on absence alone.
        - CRITICAL: The live context is a SMALL SAMPLE of recent headlines. Not finding a story in that sample is NOT evidence the story is false. Only mark "FAKE" if a live context headline DIRECTLY contradicts a specific detail (e.g. says the opposite outcome).
        - NEVER mention knowledge cutoff or lack of real-time access.
        - YEAR AWARENESS: If the LIVE WEB CONTEXT contains similar facts from a DIFFERENT year, do NOT use that to debunk the claim.
        - SELF-CHECK: Before marking FAKE, re-read your explanation. If your explanation actually confirms the key fact in the claim (same number, same outcome), change your verdict to REAL.
        
        Respond ONLY with a raw JSON object, no markdown:
        {{
            "status": "REAL" or "FAKE" or "UNVERIFIED",
            "confidence": an integer between 60 and 99,
            "explanation": "Confident factual explanation. If FAKE, cite the specific wrong detail. If UNVERIFIED: Live context was insufficient to confirm or deny this claim.",
            "source_type": "Live News Context" or "Historical Record" or "General Knowledge"
        }}
        """
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a professional fact-checker that responds only in valid JSON. Never mention knowledge cutoffs. Always give a definitive REAL or FAKE verdict."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }),
            timeout=10
        )
        
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            # OpenRouter sometimes returns content as a list of objects
            if isinstance(content, list):
                text = ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content).strip()
            else:
                text = content.strip()
            data = json.loads(text)
            
            status = str(data.get('status', '')).upper()

            # ABSENCE-OF-EVIDENCE GUARD (same logic as Groq guard)
            explanation_or = data.get('explanation', '')
            absence_phrases_or = [
                'no credible news sources report', 'no credible sources report',
                'live context does not confirm', 'no headlines', 'insufficient to confirm',
                'no evidence', 'no news sources', 'lack of any mention', 'no mention of',
                'not mentioned', 'absence of', 'context instead discusses',
                'not found in', 'no reference to', 'no coverage', 'not reported',
                'contradicted by the lack', 'lack of coverage', 'no reporting',
            ]
            contradiction_phrases_or = [
                'actually won', 'actually lost', 'actually said', 'correct figure is',
                'correct year is', 'in reality', 'in fact the', 'the real', 'true figure',
                'headline says', 'reports that', 'according to',
            ]
            if status == 'FAKE' and explanation_or:
                expl_low_or = explanation_or.lower()
                if (any(p in expl_low_or for p in absence_phrases_or) and
                        not any(p in expl_low_or for p in contradiction_phrases_or)):
                    print(f"DEBUG: OpenRouter absence-of-evidence FAKE suppressed.")
                    return None

            if status == 'UNVERIFIED':
                return None

            if data.get('confidence', 0) >= 70 and status in ['REAL', 'FAKE', 'TRUE']:
                final_status = 'real' if status in ['REAL', 'TRUE'] else 'fake'
                return (
                    final_status, 
                    float(data['confidence']), 
                    f"OpenRouter AI Verification: {data.get('explanation', 'Verified by OpenRouter AI.')}",
                    f"OpenRouter ({data.get('source_type', 'AI Knowledge')})"
                )
        else:
             print(f"OpenRouter API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"OpenRouter AI Check error: {e}")
        
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        print(f"DEBUG: /predict called with text: {text[:50]}...")

        
        # Use UI provided keys first, fallback to .env backend keys
        newsapi_key = data.get('newsapi_key') or os.environ.get('NEWSAPI_KEY')
        serpapi_key = data.get('serpapi_key') or os.environ.get('SERPAPI_KEY')
        gemini_key = data.get('gemini_api_key') or os.environ.get('GEMINI_API_KEY')
        groq_key = data.get('groq_api_key') or os.environ.get('GROQ_API_KEY')
        openrouter_key = data.get('openrouter_api_key') or os.environ.get('OPENROUTER_API_KEY')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if not vectorizer or not model:
            return jsonify({'error': 'Model not initialized'}), 500

        # Autonomous Language Detection - require at least 3 chars to avoid false positives
        tamil_chars = re.findall('[\u0B80-\u0BFF]', text)
        is_tamil = len(tamil_chars) >= 3
        
        # Translate to English for stylistic AI analysis
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(text[:500]) if is_tamil else text
        except Exception as e:
            print(f"Translation Error: {e}")
            translated_text = text

        # Preprocess
        cleaned_input = re.sub('[^a-zA-Z]', ' ', translated_text).lower()
        
        # Gibberish / Nonsensical Input Guard
        # TAMIL FIX: re.sub strips all Tamil chars → cleaned_input is empty even for valid Tamil.
        # Only reject if BOTH the cleaned ASCII version is empty AND the original has no Tamil chars.
        meaningful_words = [w for w in cleaned_input.split() if len(w) >= 2]
        has_tamil = bool(re.search('[\u0B80-\u0BFF]', text))
        if len(meaningful_words) < 2 and not has_tamil:
            return jsonify({
                'prediction': 'fake',
                'confidence': 95.0,
                'verification': 'Invalid Input',
                'message': 'No meaningful text detected. Please paste a real news headline or article.',
                'live_url': None,
                'api_error': False,
                'match_score': None,
                'matched_words': []
            })
        
        # Transform
        prediction_input = vectorizer.transform([cleaned_input])
        
        # Predict
        prediction = model.predict(prediction_input)[0]
        probability = model.predict_proba(prediction_input)[0]
        
        classes = list(model.classes_)
        
        # New model uses 0 and 1, old model used "real" and "fake"
        if 0 in classes and 1 in classes:
            real_idx = classes.index(0)
            fake_idx = classes.index(1)
            prediction = "real" if prediction == 0 else "fake"
        else:
            real_idx = classes.index("real")
            fake_idx = classes.index("fake")
        
        conf = probability[real_idx] if prediction == "real" else probability[fake_idx]
        conf = float(conf) * 100

        # Verification Logic
        verification_status = "Unverified"
        message = "Analysis based on linguistic style patterns."
        live_url = None
        api_error = False

        match = None
        
        # 1. First, check Tamil RSS if the input is Tamil
        if is_tamil:
            try:
                tamil_found, tamil_match = check_tamil_rss_sources(text)
                if tamil_found and tamil_match:
                    if check_semantic_consistency(text, tamil_match.get('title', ''), groq_key, gemini_key):
                        prediction = "real"
                        conf = max(conf, 92.0)
                        verification_status = f"Corroborated by {tamil_match['source']['name']}"
                        message = f"Verified! This headline was found in the {tamil_match['source']['name'].replace(' (Verified Source)', '')} feed."
                        live_url = tamil_match['url']
                        match = tamil_match
            except Exception as e:
                print(f"Tamil RSS check error in manual route: {e}")
        else:
            # 1b. Check English RSS/Top Headlines
            try:
                eng_found, eng_match = check_english_rss_sources(text, newsapi_key)
                if eng_found and eng_match:
                    if check_semantic_consistency(text, eng_match.get('title', ''), groq_key, gemini_key):
                        prediction = "real"
                        conf = max(conf, 92.0)
                        verification_status = f"Corroborated by {eng_match['source']['name']}"
                        message = f"Verified! This headline was found in the {eng_match['source']['name'].replace(' (Verified Source)', '')} feed."
                        live_url = eng_match['url']
                        match = eng_match
            except Exception as e:
                print(f"English RSS check error in manual route: {e}")

        # 2. AI-Powered Verification (Gemini with Google Search Grounding runs FIRST)
        #    This is the most accurate check — it searches the web in real-time.
        
        if len(text) > 20 and not live_url:
            search_query = translated_text[:100].replace('\n', ' ')
            live_context = ""    # BUG-5 FIX: initialize explicitly
            context_urls = []
            
            # STEP 2a: Gemini Search-Grounded fact-check (highest priority)
            if gemini_key:
                gemini_result = check_with_gemini(text, gemini_key)
                # FIX: Log when Gemini returns None so quota/timeout failures are visible in console
                if gemini_result is None:
                    print(f"DEBUG: Gemini returned None for query: {text[:60]!r} — falling through to Groq")
                if gemini_result:
                    gem_pred, gem_conf, gem_msg, gem_source = gemini_result[:4]
                    gem_url = gemini_result[4] if len(gemini_result) > 4 else None
                    if gem_conf >= 70:
                        prediction = gem_pred
                        conf = max(conf, gem_conf)
                        verification_status = "Verified by Google AI" if gem_pred == "real" else "Debunked by Google AI"
                        message = gem_msg
                        match = {'source': {'name': gem_source}}
                        if gem_url:
                            live_url = gem_url
            
            # STEP 2b: Groq AI fallback (if Gemini unverified/unavailable)
            if verification_status == "Unverified" and groq_key:
                live_context, context_urls = get_live_context(search_query, serpapi_key, newsapi_key)
                groq_result = check_with_groq(text, groq_key, live_context)
                if groq_result:
                    gro_pred, gro_conf, gro_msg, gro_source = groq_result
                    if gro_conf >= 70:
                        prediction = gro_pred
                        conf = max(conf, gro_conf)
                        verification_status = "Verified by Groq AI" if gro_pred == "real" else "Debunked by Groq AI"
                        message = gro_msg
                        match = {'source': {'name': gro_source}}
                        if context_urls and not live_url:
                            live_url = get_source_url_for_verdict(text, context_urls, groq_key, gemini_key, verdict=gro_pred)
            
            # STEP 2c: OpenRouter AI fallback
            if verification_status == "Unverified" and openrouter_key:
                # BUG-3 FIX: reuse already-fetched live_context instead of calling get_live_context again
                if not live_context:
                    live_context, context_urls = get_live_context(search_query, serpapi_key, newsapi_key)
                or_result = check_with_openrouter(text, openrouter_key, live_context)
                if or_result:
                    or_pred, or_conf, or_msg, or_source = or_result
                    if or_conf >= 70:
                        prediction = or_pred
                        conf = max(conf, or_conf)
                        verification_status = "Verified by OpenRouter AI" if or_pred == "real" else "Debunked by OpenRouter AI"
                        message = or_msg
                        match = {'source': {'name': or_source}}
                        if context_urls and not live_url:
                            live_url = get_source_url_for_verdict(text, context_urls, groq_key, gemini_key, verdict=or_pred)
            
            # STEP 3: Keyword-based NewsAPI/SerpAPI corroboration (only if AI checks didn't resolve it)
            if verification_status == "Unverified" and (newsapi_key or serpapi_key):
                is_fake_fc, is_real_fc, match, v_status, api_error = check_fact_checking_apis(search_query, text, newsapi_key, serpapi_key)

                if is_fake_fc:
                    prediction = "fake"
                    conf = max(conf, 95.0)
                    verification_status = "Debunked by Fact Checkers"
                    message = f"This claim has been explicitly debunked. (Source: {match['source']['name']})"
                    live_url = match['url']
                elif is_real_fc:
                    if check_semantic_consistency(text, match.get('title', ''), groq_key, gemini_key):
                        prediction = "real"
                        conf = max(conf, 95.0)
                        verification_status = "Verified by Fact Checkers"
                        message = f"This claim has been fact-checked as true. (Source: {match['source']['name']})"
                        live_url = match['url']
                    else:
                        match = None
                elif match:
                    if check_semantic_consistency(text, match.get('title', ''), groq_key, gemini_key):
                        match_source = (match.get('source') or {}).get('name', 'Trusted Source')
                        if "Corroborated" in v_status:
                            if prediction == "fake":
                                if conf < 65:
                                    message = f"Found a similar news title from {match_source}, but linguistic style remains suspicious."
                                else:
                                    prediction = "real"
                                    conf = max(conf, 85.0)
                                    message = f"Confirmed via verified news outlets! (Source: {match_source})"
                            else:
                                conf = max(conf, 90.0)
                                message = f"Verified reports from {match_source} support this story."
                        else:
                            message = f"Related topics found in general search, but analysis suggests cautionary reading. (Source: {match_source})"
                        
                        verification_status = v_status
                        live_url = match.get('url')
                    else:
                        match = None

        # BUG-4 FIX: /predict was missing the final guardrail present in /predict_image.
        # Applies the same sensationalist / sports-win downgrade for the text route.
        if len(text) > 20 and verification_status == "Unverified":
            text_lower_guard = translated_text.lower()
            high_stakes_kw = [
                'bombed', 'bombing', 'war declared', 'killed', 'massacre', 'invasion',
                'invaded', 'assassination', 'nuclear', 'airstrike', 'terrorist',
                'terrorism', 'coup', 'earthquake', 'tsunami', 'flood', 'floods',
                'cyclone', 'hurricane', 'tornado', 'volcano', 'disaster', 'outbreak',
                'pandemic', 'virus'
            ]
            sports_win_patterns = [
                r'wins?\s+(?:the\s+)?(?:icc\s+)?(?:t20|odi|test)?\s*world\s*cup',
                r'won\s+(?:the\s+)?(?:icc\s+)?(?:t20|odi|test)?\s*world\s*cup',
                r'wins?\s+(?:icc|championship|title)',
                r'beats?\s+(?:india|pakistan|england|australia|new zealand)',
                r'lifts?\s+(?:the\s+)?trophy',
                r'becomes?\s+president',
                r'wins?\s+(?:the\s+)?election',
            ]
            is_unverifiable_sport = any(re.search(p, text_lower_guard) for p in sports_win_patterns)
            is_high_stakes = any(kw in text_lower_guard for kw in high_stakes_kw)

            if is_unverifiable_sport:
                prediction = "fake"
                conf = max(conf, 80.0)
                verification_status = "Unverified"
                message = ("⚠️ This claim about a major sporting or political event could NOT be "
                           "corroborated by any live news source or AI. It is likely FABRICATED. "
                           "Do not share without verification from a trusted source.")
            elif prediction == "real" and is_high_stakes:
                verification_status = "Unverified (Sensationalist)"
                message = "Caution: This alarming claim could not be corroborated by any live news reports."
            elif prediction == "real" and conf < 55.0:
                # Genuinely low ML confidence → downgrade to FAKE
                prediction = "fake"
                verification_status = "Unverified"
                message = ("No live source or AI could corroborate this claim. "
                           "The result is based on writing style only; treat with caution.")
            elif prediction == "real" and conf < 80.0:
                # Medium ML confidence (55–80%) — do NOT flip to FAKE.
                # Writing style looks real but we have no live confirmation.
                # Show as Unverified Real so user knows to verify independently.
                conf = min(conf, 60.0)
                verification_status = "Unverified"
                message = ("Writing style appears authentic, but this claim could NOT be "
                           "confirmed by any live news source or AI. Please verify independently.")
            elif prediction == "real":
                conf = min(conf, 65.0)
                message = ("This appears real based on writing style, but could NOT be verified "
                           "by any live news or AI. Treat with caution.")

        # Smart source label based on verdict and URL type
        raw_source_name = (match.get('source') or {}).get('name') if match and isinstance(match, dict) else None
        if live_url and 'google.com/search' in str(live_url):
            display_source_name = '🔍 Search Google for this claim'
        elif not live_url and prediction == 'fake':
            display_source_name = None  # No source for fabricated claims
        else:
            display_source_name = raw_source_name

        return jsonify({
            'prediction': prediction,
            'confidence': conf,
            'verification': verification_status,
            'message': message,
            'live_url': live_url,
            'source_name': display_source_name,
            'api_error': api_error,
            'match_score': match.get('match_score') if match and isinstance(match, dict) else None,
            'matched_words': match.get('matched_words', []) if match and isinstance(match, dict) else []
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch_news')
def fetch_news():
    # Use UI provided api_key first, fallback to .env NEWSAPI_KEY
    api_key = request.args.get('api_key') or os.environ.get('NEWSAPI_KEY')
    category = request.args.get('category', 'general')
    language = request.args.get('language', 'en')
    source_param = request.args.get('source', 'newsapi')

    try:
        articles = []
        if source_param == 'english-top':
            articles = get_top_english_articles(api_key)
            if not articles and not api_key:
                return jsonify({'error': 'NewsAPI Key is required for English sources.'}), 400
        elif source_param == 'tamil-combined':
            articles = get_top_tamil_articles()
        else:
            # Standard NewsAPI logic
            if not api_key:
                return jsonify({'error': 'NewsAPI Key is required. Please set it in settings.'}), 400
            
            try:
                country = 'in' if language == 'ta' else 'us'
                url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&pageSize=100&apiKey={api_key}"
                response = requests.get(url, timeout=5)
                data = response.json()
                if data.get("status") == "ok":
                    for art in data.get("articles", []):
                        art['is_tamil'] = (language == 'ta')
                        articles.append(art)
                else:
                    return jsonify({'error': data.get('message', 'Failed to fetch news')}), 400
            except Exception as e:
                print(f"NewsAPI error: {e}")

        articles = articles[:100]
        processed_articles = []
        
        for article in articles:
            headline = article.get('title', '')
            if not headline:
                continue
                
            source_obj = article.get('source', {})
            source_name = source_obj.get('name', 'Unknown') if isinstance(source_obj, dict) else str(source_obj or 'Unknown')
            
            trusted_sources = [
                # Tamil sources
                "BBC Tamil", "Polimer News", "News 7 Tamil",
                "Daily Thanthi", "DailyThanthi", "Puthiya Thalamurai",
                "Sun News", "Thanthi TV", "NDTV Tamil",
                "Dinamalar", "Dinamani", "Vikatan", "Maalai Malar",
                "OneIndia Tamil", "Tamil Murasu", "Kalaignar TV", "Nakkheeran",
                # English — global wire services
                "Reuters", "Reuters Top", "AP News", "Associated Press",
                # English — international broadcasters & papers
                "BBC News", "BBC UK", "Al Jazeera", "The Guardian",
                "NPR News", "Washington Post",
                "The New York Times", "The Wall Street Journal",
                # CNN
                "CNN", "CNN World", "CNN US", "CNN Politics", "CNN Business",
                # English — India
                "The Hindu", "The Indian Express", "Times of India",
                "NDTV", "Hindustan Times", "India Today", "Economic Times",
                "Deccan Herald", "News18", "The Wire", "Scroll.in",
                "ABC News",
            ]
            
            # Trusted Source Bypass
            if any(trusted.lower() in source_name.lower() for trusted in trusted_sources):
                prediction = "real"
                conf = 0.99
            else:
                # Clean title (remove channel suffix often found in Google News RSS)
                cleaned_hl = headline
                for trusted in trusted_sources:
                    cleaned_hl = re.sub(f" - {trusted}$", "", cleaned_hl)

                # Detect and Translate Tamil for stylistic analysis
                is_tamil = bool(re.search('[\u0B80-\u0BFF]', cleaned_hl))
                analysis_text = cleaned_hl
                if is_tamil:
                    try:
                        analysis_text = GoogleTranslator(source='auto', target='en').translate(cleaned_hl[:500])
                    except Exception as e:
                        print(f"Translation error in news feed: {e}")
                
                # Stylistic AI Analysis
                final_cleaned = re.sub('[^a-zA-Z]', ' ', analysis_text).lower()
                pred_input = vectorizer.transform([final_cleaned])
                
                classes = list(model.classes_)
                raw_pred = model.predict(pred_input)[0]
                probability = model.predict_proba(pred_input)[0]
                
                if 0 in classes and 1 in classes:
                    real_idx = classes.index(0)
                    fake_idx = classes.index(1)
                    prediction = "real" if raw_pred == 0 else "fake"
                else:
                    real_idx = classes.index("real")
                    fake_idx = classes.index("fake")
                    prediction = raw_pred
                
                conf = probability[real_idx] if prediction == "real" else probability[fake_idx]

            # Translation for display if needed
            final_display_title = headline
            if language == 'ta' and not any(trusted in source_name for trusted in trusted_sources):
                try:
                    final_display_title = GoogleTranslator(source='auto', target='ta').translate(headline)
                except Exception as e:
                    print(f"Display Translation error: {e}")
            
            processed_articles.append({
                'title': final_display_title,
                'source': source_name,
                'url': article.get('url'),
                'image': article.get('urlToImage'),
                'prediction': prediction,
                'confidence': float(conf) * 100
            })
            
        return jsonify({'articles': processed_articles})
        
    except Exception as e:
        print(f"Fetch news error: {e}")
        return jsonify({'error': str(e)}), 500

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')  # BUG-7 FIX: configurable via .env

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def ocr_denoise_and_extract_claim(raw_text, groq_key=None, gemini_key=None):
    """
    Uses an LLM to clean up messy OCR text and extract the most likely news headline or core claim.
    Returns (cleaned_headline, is_likely_news)
    """
    if not raw_text or len(raw_text.strip()) < 10:
        return raw_text, False

    prompt = f"""
    Analyze the following raw OCR text extracted from an image. 
    1. Clean up any OCR errors (typos, misread characters).
    2. Identify if this text contains a news headline, a social media claim, or a breaking news alert.
    3. Extract ONLY the primary news headline or claim. 
    4. If the text is just gibberish, UI elements, or not a news claim, return "NO_CLAIM_FOUND".

    RAW OCR TEXT:
    {raw_text}

    Respond ONLY with a JSON object:
    {{
        "cleaned_headline": "The extracted headline or claim here",
        "is_likely_news": true/false,
        "language": "en" or "ta"
    }}
    """

    # Try Groq first as it's fast
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            headline = data.get('cleaned_headline', '').strip()
            if headline and headline != "NO_CLAIM_FOUND":
                return headline, data.get('is_likely_news', False)
        except Exception as e:
            print(f"OCR Denoise (Groq) error: {e}")

    # Fallback to Gemini if Groq fails or is unavailable
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            data = json.loads(response.text)
            headline = data.get('cleaned_headline', '').strip()
            if headline and headline != "NO_CLAIM_FOUND":
                return headline, data.get('is_likely_news', False)
        except Exception as e:
            print(f"OCR Denoise (Gemini) error: {e}")

    return raw_text, True # Default to original if LLM fails


def clean_ocr_text(text):
    """
    Fixes common Tesseract OCR misreads before passing text to AI/search.
    Key fix: Tesseract misreads the Rs symbol as 2, so Rs.131 crore becomes 2131 crore.
    Corrects this while preserving real numbers like Rs. 2500 crore.
    """
    if not text:
        return text

    RUPEE = '₹'  # actual ₹ char — avoids \u escape issues in regex replacements

    def fix_rupee_misread(m):
        # Skip if preceded by Rs., INR, or another digit (it is a real number)
        prefix = text[:m.start()]
        if re.search(r'(?:rs\.?\s*|inr\s*|\d)$', prefix, re.IGNORECASE):
            return m.group(0)
        return f'{RUPEE}{m.group(1)} {m.group(2)}'

    # BUG-C FIX: Previously \d{2,3} matched legitimate 4-digit numbers (2045 crore →
    # ₹045 crore, 2100 crore → ₹100 crore).  Restrict to \d{2} only so we only catch
    # 3-digit totals (e.g. ₹31 misread as 231).  The ₹131 → 2131 (4-digit) case cannot
    # be safely auto-corrected without broader context and is left to the LLM denoiser.
    text = re.sub(
        r'\b2(\d{2})\s+(crore|lakh|rupee|million|billion|thousand)\b',
        fix_rupee_misread,
        text,
        flags=re.IGNORECASE
    )
    # Fix: yen symbol (¥) misread instead of rupee — replace with actual ₹ char
    text = re.sub(r'(?<!\d)¥\s*(\d)', RUPEE + r'\1', text)
    # Fix: letter O misread as 0 inside numbers
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    # Fix: l or I misread as 1 inside numbers
    text = re.sub(r'(?<=\d)[lI](?=\d)', '1', text)
    return text


@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    # Use UI provided keys first, fallback to .env backend keys
    newsapi_key = request.form.get('newsapi_key') or os.environ.get('NEWSAPI_KEY')
    serpapi_key = request.form.get('serpapi_key') or os.environ.get('SERPAPI_KEY')
    gemini_key = request.form.get('gemini_api_key') or os.environ.get('GEMINI_API_KEY')
    groq_key = request.form.get('groq_api_key') or os.environ.get('GROQ_API_KEY')
    openrouter_key = request.form.get('openrouter_api_key') or os.environ.get('OPENROUTER_API_KEY')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            print(f"DEBUG: /predict_image called with file: {file.filename}")

            # 1. Extract text from the image using OCR (English + Tamil)
            pil_img = Image.open(filepath)
            extracted_text = pytesseract.image_to_string(pil_img, lang='eng+tam').strip()
            extracted_text = clean_ocr_text(extracted_text)  # Fix common OCR symbol misreads (₹→2, etc.)
            print(f"DEBUG: OCR extracted (after cleanup): {extracted_text[:100]}")

            if not extracted_text:
                return jsonify({
                    'error': 'No text could be extracted from this image. '
                             'Please upload a clear screenshot of a news article or headline.'
                }), 400

            # Autonomous Language Detection - require at least 3 chars to avoid OCR hallucinations
            tamil_chars = re.findall('[\u0B80-\u0BFF]', extracted_text)
            is_tamil = len(tamil_chars) >= 3

            # 1b. Denoise and extract core claim using LLM
            print("DEBUG: Denoising OCR text...")
            denoised_text, is_likely_news = ocr_denoise_and_extract_claim(
                extracted_text, 
                groq_key=groq_key, 
                gemini_key=gemini_key
            )
            
            if not is_likely_news and len(extracted_text) > 20:
                 print(f"DEBUG: OCR deemed unlikely to be news: {denoised_text}")
            
            search_text = denoised_text if denoised_text else extracted_text
            print(f"DEBUG: Proceeding with search text: {search_text}")

            if not vectorizer or not model:
                return jsonify({'error': 'Model not initialized'}), 500
            
            # Translate text to English for the AI model
            try:
                translated_text = GoogleTranslator(source='auto', target='en').translate(search_text) if is_tamil else search_text
            except Exception as e:
                print(f"Translation Error: {e}")
                translated_text = search_text

            # 2. Run text through the AI stylistic classifier using translated text
            cleaned = re.sub('[^a-zA-Z]', ' ', translated_text).lower()
            
            # Gibberish / Nonsensical Input Guard
            # TAMIL FIX: same as /predict — don't reject valid Tamil text that strips to empty ASCII
            meaningful_words = [w for w in cleaned.split() if len(w) >= 2]
            has_tamil_img = bool(re.search('[\u0B80-\u0BFF]', search_text))
            if len(meaningful_words) < 2 and not has_tamil_img:
                return jsonify({
                    'prediction': 'fake',
                    'confidence': 95.0,
                    'verification': 'Invalid Input',
                    'message': 'No meaningful text detected in the image. Please upload a clear news screenshot.',
                    'extracted_text': extracted_text,
                    'live_url': None,
                    'api_error': False,
                    'match_score': None,
                    'matched_words': []
                })
            
            pred_input = vectorizer.transform([cleaned])
            prediction = model.predict(pred_input)[0]
            probability = model.predict_proba(pred_input)[0]

            classes = list(model.classes_)
            if 0 in classes and 1 in classes:
                real_idx = classes.index(0)
                fake_idx = classes.index(1)
                prediction = "real" if prediction == 0 else "fake"
            else:
                real_idx = classes.index("real")
                fake_idx = classes.index("fake")
            
            conf = (probability[real_idx] if prediction == "real" else probability[fake_idx]) * 100

            # 3. Local Heuristic for Unprofessional Language
            verification_status = "Unverified"
            message = "Analysis based on linguistic style patterns."
            live_match = None
            
            # Check for unprofessional language (simple heuristic for satire/fake)
            unprofessional_words = ['shit', 'fuck', 'damn', 'sucks', 'stupid', 'idiot']
            is_unprofessional = any(word in cleaned.split() for word in unprofessional_words)

            if is_unprofessional and prediction == "real":
                prediction = "fake"
                conf = max(conf, 85.0)
                message = "Flagged due to use of highly unprofessional or explicit language in a news context."

            word_count = len(translated_text.split())
            final_label = prediction.upper()

            # 4. Verification Check for Images (Matches Live News)
            # Use original extracted text for BBC check, translated for others
            # 4a. Tamil Channels RSS check (runs first, before NewsAPI/SerpAPI)
            if is_tamil:
                try:
                    tamil_found, tamil_match = check_tamil_rss_sources(extracted_text)
                    if tamil_found and tamil_match:
                        if check_semantic_consistency(extracted_text, tamil_match.get('title', ''), groq_key):
                            prediction = "real"
                            final_label = "REAL"
                            conf = max(conf, 92.0)
                            verification_status = f"Corroborated by {tamil_match['source']['name']}"
                            message = f"Verified! This headline was found in the {tamil_match['source']['name'].replace(' (Verified Source)', '')} feed."
                            live_match = tamil_match
                except Exception as e:
                    print(f"Tamil RSS check error in image route: {e}")
            else:
                # 4b. English Channels RSS check
                try:
                    eng_found, eng_match = check_english_rss_sources(extracted_text, newsapi_key)
                    if eng_found and eng_match:
                        if check_semantic_consistency(extracted_text, eng_match.get('title', ''), groq_key):
                            prediction = "real"
                            final_label = "REAL"
                            conf = max(conf, 92.0)
                            verification_status = f"Corroborated by {eng_match['source']['name']}"
                            message = f"Verified! This headline was found in the {eng_match['source']['name'].replace(' (Verified Source)', '')} feed."
                            live_match = eng_match
                except Exception as e:
                    print(f"English RSS check error in image route: {e}")

            if (newsapi_key or serpapi_key) and not live_match:
                try:
                    search_query = translated_text[:100].replace('\n', ' ')
                    # FIX: function returns 5 values (is_fake, is_real, match, status, api_error)
                    _, is_real_fc, match, v_status, _ = check_fact_checking_apis(search_query, extracted_text, newsapi_key, serpapi_key)
                    
                    if is_real_fc:
                        if check_semantic_consistency(extracted_text, match.get('title', ''), groq_key):
                            prediction = "real"
                            final_label = "REAL"
                            conf = max(conf, 95.0)
                            verification_status = v_status
                            message = f"This claim has been fact-checked as true. (Source: Fact Checker)"
                            live_match = match
                    elif match:
                        if check_semantic_consistency(extracted_text, match.get('title', ''), groq_key):
                            match_source = (match.get('source') or {}).get('name', 'Trusted Source')
                            if "Corroborated" in v_status:
                                if prediction == "fake":
                                    if conf < 65: # AI is very sure it's fake
                                        message = f"Found a similar news title from {match_source}, but linguistic style remains suspicious."
                                    else:
                                        prediction = "real"
                                        final_label = "REAL"
                                        conf = max(conf, 90.0)
                                        message = f"Confirmed via verified news outlets! (Source: {match_source})"
                                else:
                                    conf = max(conf, 90.0)
                                    message = f"Verified reports from {match_source} support this story."
                            else:
                                # Related reports found but not high enough confidence to override "FAKE" stylistics
                                message = f"Related news found, but analysis suggests caution. (Source: {match_source})"
                            
                            verification_status = v_status
                            live_match = match
                except Exception as e:
                    print(f"Image Fact-Check API error: {e}")

            # 5. SENSATIONALIST GUARDRAILS & AI FALLBACKS
            # Only run AI fallbacks if we haven't already confirmed via a live source
            if verification_status == "Unverified":
                search_query = search_text[:100].replace('\n', ' ')
                
                # STEP 5a: Gemini Search-Grounded check (FIRST)
                if gemini_key:
                    gemini_result = check_with_gemini(search_query, gemini_key)
                    # FIX: Log Gemini None so quota/timeout failures are visible
                    if gemini_result is None:
                        print(f"DEBUG [image]: Gemini returned None for: {search_query[:60]!r}")
                    if gemini_result:
                        gem_pred, gem_conf, gem_msg, gem_source = gemini_result[:4]
                        gem_url = gemini_result[4] if len(gemini_result) > 4 else None
                        prediction = gem_pred
                        final_label = gem_pred.upper()
                        conf = max(conf, gem_conf)
                        verification_status = "Verified by Google AI" if gem_pred == "real" else "Debunked by Google AI"
                        message = gem_msg
                        live_match = {'source': {'name': gem_source}}
                        if gem_url: live_match['url'] = gem_url
                
                # STEP 5b: Groq AI fallback
                if "context_urls_img" not in locals():
                    context_urls_img = []  # BUG-6 FIX: ensure always defined
                if verification_status == "Unverified" and groq_key:
                    live_context_img, context_urls_img = get_live_context(search_query, serpapi_key, newsapi_key)
                    groq_result = check_with_groq(search_query, groq_key, live_context_img)
                    if groq_result:
                        gro_pred, gro_conf, gro_msg, gro_source = groq_result
                        prediction = gro_pred
                        final_label = gro_pred.upper()
                        conf = max(conf, gro_conf)
                        verification_status = "Verified by Groq AI" if gro_pred == "real" else "Debunked by Groq AI"
                        message = gro_msg
                        live_match = {'source': {'name': gro_source}, 'url': get_source_url_for_verdict(search_query, context_urls_img, groq_key, gemini_key, verdict=gro_pred)}

                # STEP 5c: OpenRouter AI fallback
                if verification_status == "Unverified" and openrouter_key:
                    if 'context_urls_img' not in locals():
                        _, context_urls_img = get_live_context(search_query, serpapi_key, newsapi_key)
                    or_result = check_with_openrouter(search_query, openrouter_key)
                    if or_result:
                        or_pred, or_conf, or_msg, or_source = or_result
                        prediction = or_pred
                        final_label = or_pred.upper()
                        conf = max(conf, or_conf)
                        verification_status = "Verified by OpenRouter AI" if or_pred == "real" else "Debunked by OpenRouter AI"
                        message = or_msg
                        live_match = {'source': {'name': or_source}, 'url': get_source_url_for_verdict(search_query, context_urls_img if 'context_urls_img' in locals() else [], groq_key, gemini_key, verdict=or_pred)}
                
                try:
                    if verification_status == "Unverified":
                        high_stakes_keywords = [
                            'bombed', 'bombing', 'war declared', 'killed', 'massacre', 'invasion', 
                            'invaded', 'assassination', 'nuclear', 'airstrike', 'terrorist', 
                            'terrorism', 'coup', 'earthquake', 'tsunami', 'flood', 'floods', 
                            'cyclone', 'hurricane', 'tornado', 'volcano', 'disaster', 'outbreak', 
                            'pandemic', 'virus'
                        ]
                        is_high_stakes = any(word in (translated_text.lower()) for word in high_stakes_keywords)

                        if prediction == "real" and is_high_stakes:
                            verification_status = "Unverified (Sensationalist)"
                            message = "Caution: This alarming claim could not be corroborated by any live news reports."
                
                        # --- STRONG FINAL GUARDRAIL ---
                        # If prediction is still "real" after ALL verification steps failed (no AI, no live source),
                        # we must not trust the raw ML model output alone.
                        # Sports/championship wins and similar specific event claims are a very common
                        # fake news pattern and must be caught here.
                        if prediction == "real" and verification_status == "Unverified":
                            TODAY_YEAR = datetime.now().year
                            text_lower = translated_text.lower()
                            
                            # Detect specific unverifiable sporting events, election wins, etc.
                            sports_win_patterns = [
                                'wins world cup', 'won world cup', 'win world cup',
                                'wins icc', 'won icc', 'beats india', 'beats pakistan',
                                'beats england', 'beats australia', 'wins championship',
                                'wins title', 'becomes president', 'wins election',
                                'lifts trophy', 'lifts the trophy'
                            ]
                            is_unverifiable_sport = any(p in text_lower for p in sports_win_patterns)
                            
                            if is_unverifiable_sport:
                                prediction = "fake"
                                final_label = "FAKE"
                                conf = max(conf, 80.0)
                                verification_status = "Unverified"
                                message = "⚠️ This claim about a major sporting or political event could NOT be corroborated by any live news source or AI. It is likely FABRICATED. Do not share without verification from a trusted source."
                            elif conf < 55.0:
                                # Genuinely low ML confidence → downgrade to FAKE
                                prediction = "fake"
                                final_label = "FAKE"
                                verification_status = "Unverified"
                                message = "No live source or AI could corroborate this claim. The result is based on writing style only; treat with caution."
                            elif conf < 80.0:
                                # Medium ML confidence (55–80%) — do NOT flip to FAKE.
                                # Writing style looks real but no live confirmation.
                                conf = min(conf, 60.0)
                                verification_status = "Unverified"
                                message = ("Writing style appears authentic, but this claim could NOT be "
                                           "confirmed by any live news source or AI. Please verify independently.")
                            else:
                                # High-confidence unverified real → soften (never show 94% real with no source)
                                conf = min(conf, 65.0)
                                verification_status = "Unverified"
                                message = "This appears real based on writing style, but could NOT be verified by any live news or AI. Treat with caution."

                except Exception as e:
                    print(f"Image Verification Error: {e}")

            img_live_url = live_match.get('url') if live_match and isinstance(live_match, dict) else (live_match if isinstance(live_match, str) else None)
            # BUG-E FIX: use (or {}) to handle source=None; simplify the overly complex guard
            img_raw_source = (live_match.get('source') or {}).get('name') if live_match and isinstance(live_match, dict) else None
            if img_live_url and 'google.com/search' in str(img_live_url):
                img_source_name = '🔍 Search Google for this claim'
            elif not img_live_url and final_label == 'FAKE':
                img_source_name = None  # No source for fabricated claims
            else:
                img_source_name = img_raw_source

            return jsonify({
                'original': f"/static/uploads/{filename}",
                'prediction': final_label,
                'confidence': float(conf),
                'extracted_text': extracted_text[:500],
                'translated_text': translated_text[:500] if translated_text != extracted_text else None,
                'word_count': word_count,
                'verification': verification_status,
                'message': message,
                'live_url': img_live_url,
                'source_name': img_source_name,
                'api_error': False  # BUG-5 FIX: field was absent; frontend reads this on both routes
            })

        except Exception as e:
            print(f"Image OCR/Analysis Error: {e}")
            return jsonify({'error': f'Analysis Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
