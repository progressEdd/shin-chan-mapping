from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class MatchedTitle(BaseModel):
    mapped_title: str = Field(..., description="the full title among the list that is the most accurate")
    reason: str = Field(..., description="reason for why the title was picked over the rest")
    negatives: Dict[str, str] = Field(..., description="Reasons why the other candidates were not chosen")

from functools import wraps
def retry_until_valid(func):
    @wraps(func)                       # keep metadata
    def wrapper(*args, **kwargs):      # forward any signature
        attempt = 1
        while True:
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                print(f"Attempt #{attempt} failed: {e}")
                attempt += 1           # loop back → new LLM call
    return wrapper  

@retry_until_valid
def llm_matching(title_to_map, title_language, retrieved_titles, retrived_title_language, llm_model="gemma3:27b"):
    json_completion = client.beta.chat.completions.parse(
        model="gemma3:27b",

        response_format= MatchedTitle,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in German–Japanese–English translation and title matching.\n"
                    "Your task: given one source title and a list of candidate titles, select the single best match.\n\n"
                    "Rules:\n"
                    "1) Only choose from the provided list — never invent or alter a title.\n"
                    "2) Prefer the candidate that matches the meaning and context of the Japanese title, "
                    "while also aligning with the English.\n"
                    "3) Normalize for comparison: ignore case, punctuation, quotes, "
                    "season/episode numbers (unless they help disambiguate), and small spelling variations.\n"
                    "4) If multiple candidates are close, prefer:\n"
                    "   a) The more specific or exact subtitle.\n"
                    "   b) The one preserving unique names or key terms.\n"
                    "5) Return:\n"
                    "   - The chosen candidate string exactly as given\n"
                    "   - A brief reason for why it was selected\n"
                    "   - Brief reasons why the other candidates are less suitable"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Source ({title_language}) title:\n{title_to_map}\n\n"
                    f"Candidate ({retrieved_title_language}) titles:\n{retrieved_titles}"
                )
            }
        ],
        temperature=0.02,
        top_p=0.1
        )
        
    MatchedTitle.model_validate(json_completion.choices[0].message.parsed)
    return json_completion.choices[0].message.parsed.model_dump()    