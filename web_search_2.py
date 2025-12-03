import os
import time
import requests
from duckduckgo_search import DDGS

from dotenv import load_dotenv
import os

load_dotenv()

def serper_search(query: str, api_key: str, num_results: int = 5) -> list[dict]:
    """
    Search using Serper.dev API
    
    Args:
        query: Search query string
        api_key: Serper.dev API key
        num_results: Number of results to return (default 5)
    
    Returns:
        List of dicts with 'title', 'link', 'snippet' keys
    """
    url = "https://google.serper.dev/search"
    
    payload = {
        "q": query,
        "num": num_results
    }
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Extract organic results
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        
        return results
    
    except requests.exceptions.Timeout:
        print(f"⚠️  Serper timeout for query: {query}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Serper error for query '{query}': {e}")
        return []
    except Exception as e:
        print(f"⚠️  Unexpected Serper error: {e}")
        return []


def duckduckgo_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Search using DuckDuckGo as fallback
    
    Args:
        query: Search query string
        num_results: Number of results to return (default 5)
    
    Returns:
        List of dicts with 'title', 'link', 'snippet' keys
    """
    try:
        ddgs = DDGS()
        results = []
        
        # Use text search with max_results
        search_results = ddgs.text(query, max_results=num_results)
        
        for item in search_results:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("href", ""),
                "snippet": item.get("body", "")
            })
        
        return results
    
    except Exception as e:
        print(f"⚠️  DuckDuckGo error for query '{query}': {e}")
        return []


def hybrid_search(query: str, serper_api_key: str, num_results: int = 5) -> list[dict]:
    """
    Hybrid search: Try Serper first, fallback to DuckDuckGo
    
    Args:
        query: Search query string
        serper_api_key: Serper.dev API key
        num_results: Number of results to return
    
    Returns:
        List of search results
    """
    print(f"🔍 Searching: {query}")
    
    # Try Serper first
    results = serper_search(query, serper_api_key, num_results)
    
    if results:
        print(f"✅ Serper returned {len(results)} results")
        return results
    
    # Fallback to DuckDuckGo
    print("🦆 Falling back to DuckDuckGo...")
    results = duckduckgo_search(query, num_results)
    
    if results:
        print(f"✅ DuckDuckGo returned {len(results)} results")
    else:
        print("❌ Both search methods failed")
    
    return results


def extract_content_from_results(results: list[dict]) -> str:
    """
    Extract and combine content from search results (mimics your Tavily logic)
    
    Args:
        results: List of search result dicts
    
    Returns:
        Combined content string
    """
    contents = []
    
    for item in results:
        # Combine title and snippet for richer context
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        
        if snippet:
            content = f"{title}: {snippet}" if title else snippet
            contents.append(content)
    
    return "\n\n".join(contents)


def web_search_hybrid(questions: list[str], serper_api_key: str) -> list[str]:
    """
    Main function that matches your original web_search signature
    
    Args:
        questions: List of search questions
        serper_api_key: Serper.dev API key
    
    Returns:
        List of combined content strings (one per question)
    """
    raw_evidence = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}")
        print(f"{'='*60}")
        
        # Get search results
        results = hybrid_search(question, serper_api_key, num_results=5)
        
        # Extract content
        combined = extract_content_from_results(results)
        raw_evidence.append(combined)
        
        # Rate limiting - be nice to the APIs
        if i < len(questions):
            time.sleep(1)
    
    return raw_evidence


# ============================================================================
# TEST CODE - Run this to verify everything works
# ============================================================================

if __name__ == "__main__":
    # Replace with your actual Serper API key
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    
    # Test questions
    test_questions = [
        "What is the capital of France?",
        "Who won the 2024 NBA championship?",
        "What are the main features of Python 3.12?"
    ]   
    
    print("🚀 Starting Hybrid Search Test\n")
    
    # Run the search
    evidence = web_search_hybrid(test_questions, SERPER_API_KEY)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for i, (question, content) in enumerate(zip(test_questions, evidence), 1):
        print(f"\n📝 Question {i}: {question}")
        print(f"📊 Content length: {len(content)} characters")
        print(f"Preview: {content[:200]}...")
        print("-" * 60)
    
    print("\n✅ Test completed!")