#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query reformulation for medical retrieval.

Transforms clinical vignette questions into search-optimized queries.
"""

QUERY_REFORMULATION_PROMPT = """You are a medical information retrieval specialist. Your task is to transform clinical vignette questions into optimized search queries for retrieving relevant medical textbook passages.

**Input**: A clinical vignette question (typically from USMLE/medical board exams)

**Your Task**:
1. Extract the core medical concepts and conditions
2. Identify the key question being asked
3. Remove irrelevant patient details (age, gender, backstory, social context)
4. Create a concise, search-optimized query that would match textbook headings/passages

**Rules**:
- Preserve medical terminology exactly as written
- Include synonyms/related terms that might appear in textbooks
- Focus on the PATHOPHYSIOLOGY or MECHANISM being tested
- Remove numbers, ages, names, timeline details unless medically relevant
- Output 2-3 variations: one specific, one broader, one mechanism-focused
- Keep each query under 15 words
- Use medical terminology, not lay language

**Example**:

INPUT QUESTION:
"A 17-year-old female accidentally eats a granola bar manufactured on equipment that processes peanuts. She develops type I hypersensitivity-mediated histamine release, resulting in pruritic wheals on the skin. Which of the following layers of this patient's skin would demonstrate histologic changes on biopsy of her lesions?"

OUTPUT (JSON format):
```json
{
  "specific_query": "peanut allergy skin histology epidermis dermis urticaria type I hypersensitivity",
  "broad_query": "type I hypersensitivity skin layers histologic changes urticaria wheals",
  "mechanism_query": "IgE mediated mast cell degranulation skin histology urticaria pathophysiology",
  "key_concepts": ["peanut allergy", "type I hypersensitivity", "urticaria", "skin histology", "epidermis", "dermis", "IgE", "mast cells"],
  "removed_details": ["17-year-old female", "granola bar", "accidentally eats", "equipment that processes peanuts"]
}
```

**Another Example**:

INPUT QUESTION:
"A 38-year-old woman comes to the physician because of difficulty falling asleep for the past 2 months. She wakes up frequently during the night and gets up earlier than desired. She experiences discomfort in her legs when lying down at night and feels the urge to move her legs. The discomfort resolves when she gets up and walks around or moves her legs. She has tried an over-the-counter sleep aid that contains diphenhydramine, which worsened her symptoms. She exercises regularly and eats a well-balanced diet. She admits that she has been under a lot of stress lately. Her brother has similar symptoms. The patient appears anxious. Physical examination shows no abnormalities. A complete blood count and iron studies are within the reference range. Which of the following is the most appropriate pharmacotherapy for this patient's symptoms?"

OUTPUT (JSON format):
```json
{
  "specific_query": "restless legs syndrome RLS treatment dopamine agonists pramipexole ropinirole",
  "broad_query": "restless legs syndrome pharmacotherapy treatment dopamine agonists gabapentin",
  "mechanism_query": "RLS pathophysiology dopamine iron deficiency basal ganglia",
  "key_concepts": ["restless legs syndrome", "RLS", "dopamine agonists", "pramipexole", "ropinirole", "gabapentin", "iron deficiency", "periodic limb movement"],
  "removed_details": ["38-year-old woman", "2 months", "over-the-counter sleep aid", "diphenhydramine", "exercises regularly", "stress", "brother", "anxious", "normal physical exam"]
}
```

Now, transform the following medical question into optimized search queries:

{question}

Respond ONLY with valid JSON in the format shown above.
"""


def parse_reformulation_response(response: str) -> dict:
    """
    Parse LLM response into structured query reformulations.

    Args:
        response: Raw LLM response text

    Returns:
        Dictionary with reformulated queries and metadata
    """
    import json
    import re

    # Try to extract JSON from response
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        response = json_match.group(1)

    # Remove any markdown code blocks
    response = re.sub(r'```\s*', '', response)

    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # Fallback: try to extract key information
        return {
            "specific_query": response[:500],
            "broad_query": "",
            "mechanism_query": "",
            "key_concepts": [],
            "removed_details": [],
            "parse_error": True
        }


def reformulate_query_with_llm(
    question: str,
    llm_func=None,
    model: str = "deepseek-chat"
) -> dict:
    """
    Reformulate a medical query using an LLM.

    Args:
        question: Original clinical vignette question
        llm_func: Function to call LLM (e.g., openrouter_client.chat.completions.create)
                 If None, uses default deepseek setup
        model: Model name to use

    Returns:
        Dictionary with reformulated queries and metadata
    """
    from jinja2 import Template

    # Use template
    prompt = QUERY_REFORMULATION_PROMPT.format(question=question)

    # Call LLM
    if llm_func is None:
        # Default: use OpenRouter with deepseek
        from openrouter import OpenRouter
        client = OpenRouter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical information retrieval specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=500
        )
        response_text = response.choices[0].message.content
    else:
        response_text = llm_func(prompt)

    # Parse response
    result = parse_reformulation_response(response_text)
    result["original_question"] = question

    return result


def get_retrieval_queries(reformulation: dict, strategy: str = "specific") -> str:
    """
    Get the best query for retrieval based on strategy.

    Args:
        reformulation: Output from reformulate_query_with_llm
        strategy: Which query to use ("specific", "broad", "mechanism", "combined")

    Returns:
        Reformulated query string for retrieval
    """
    if strategy == "specific":
        return reformulation.get("specific_query", "")
    elif strategy == "broad":
        return reformulation.get("broad_query", "")
    elif strategy == "mechanism":
        return reformulation.get("mechanism_query", "")
    elif strategy == "combined":
        # Combine all three for comprehensive search
        queries = [
            reformulation.get("specific_query", ""),
            reformulation.get("broad_query", ""),
            reformulation.get("mechanism_query", "")
        ]
        return " ".join([q for q in queries if q])
    elif strategy == "concepts":
        # Just the key concepts
        concepts = reformulation.get("key_concepts", [])
        return " ".join(concepts)
    else:
        return reformulation.get("specific_query", "")


# Example usage
if __name__ == "__main__":
    # Test with sample question
    sample_question = """A 17-year-old female accidentally eats a granola bar manufactured on equipment that processes peanuts. She develops type I hypersensitivity-mediated histamine release, resulting in pruritic wheals on the skin. Which of the following layers of this patient's skin would demonstrate histologic changes on biopsy of her lesions?"""

    print("="*100)
    print("QUERY REFORMULATION TEST")
    print("="*100)
    print(f"\nOriginal Question:\n{sample_question}\n")

    # Simulate LLM response (for testing without API)
    mock_llm_response = """```json
{
  "specific_query": "peanut allergy skin histology epidermis dermis urticaria type I hypersensitivity",
  "broad_query": "type I hypersensitivity skin layers histologic changes urticaria wheals",
  "mechanism_query": "IgE mediated mast cell degranulation skin histology urticaria pathophysiology",
  "key_concepts": ["peanut allergy", "type I hypersensitivity", "urticaria", "skin histology", "epidermis", "dermis", "IgE", "mast cells"],
  "removed_details": ["17-year-old female", "granola bar", "accidentally eats", "equipment that processes peanuts"]
}
```"""

    result = parse_reformulation_response(mock_llm_response)
    result["original_question"] = sample_question

    print("\nReformulated Queries:")
    print(f"  Specific:   {result['specific_query']}")
    print(f"  Broad:      {result['broad_query']}")
    print(f"  Mechanism:  {result['mechanism_query']}")
    print(f"  Concepts:   {', '.join(result['key_concepts'])}")

    print(f"\nRemoved Details: {', '.join(result['removed_details'])}")

    print("\n" + "="*100)
    print("RETRIEVAL STRATEGIES")
    print("="*100)

    for strategy in ["specific", "broad", "mechanism", "combined", "concepts"]:
        query = get_retrieval_queries(result, strategy=strategy)
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  {query}")
