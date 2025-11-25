#!/usr/bin/env python3
"""Test script to verify API keys for Anthropic, OpenAI, and Google."""

import os
import sys
from pathlib import Path

# Add project root to path to import config
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from human_ttc_eval.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY


def test_anthropic():
    """Test Anthropic API key."""
    print("Testing ANTHROPIC_API_KEY...")
    if not ANTHROPIC_API_KEY:
        print("  ❌ ANTHROPIC_API_KEY not set")
        return False
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'test'"}]
        )
        print(f"  ✅ ANTHROPIC_API_KEY is valid")
        print(f"     Response: {response.content[0].text[:50]}")
        return True
    except Exception as e:
        print(f"  ❌ ANTHROPIC_API_KEY test failed: {e}")
        return False


def test_openai():
    """Test OpenAI API key."""
    print("Testing OPENAI_API_KEY...")
    if not OPENAI_API_KEY:
        print("  ❌ OPENAI_API_KEY not set")
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'test'"}]
        )
        print(f"  ✅ OPENAI_API_KEY is valid")
        print(f"     Response: {response.choices[0].message.content[:50]}")
        return True
    except Exception as e:
        print(f"  ❌ OPENAI_API_KEY test failed: {e}")
        return False


def test_google():
    """Test Google API key."""
    print("Testing GOOGLE_API_KEY...")
    if not GOOGLE_API_KEY:
        print("  ❌ GOOGLE_API_KEY not set")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Say 'test'")
        print(f"  ✅ GOOGLE_API_KEY is valid")
        print(f"     Response: {response.text[:50]}")
        return True
    except Exception as e:
        print(f"  ❌ GOOGLE_API_KEY test failed: {e}")
        return False


def main():
    """Run all API key tests."""
    print("=" * 60)
    print("API Key Test Script")
    print("=" * 60)
    print()
    
    results = {
        "Anthropic": test_anthropic(),
        "OpenAI": test_openai(),
        "Google": test_google(),
    }
    
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    for provider, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {provider}: {status}")
    
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()


