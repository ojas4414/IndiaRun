import re

class InjectionDefender:
    def __init__(self):
        # Zero-width spaces, non-joiners, etc.
        self.zero_width_pattern = re.compile(r'[\u200B-\u200D\uFEFF]')
        # Basic HTML tag stripping
        self.html_pattern = re.compile(r'<[^>]*>')
        # Suspicious phrases
        self.injection_phrases = [
            r"ignore previous",
            r"you must rank",
            r"system prompt",
            r"forget all instructions"
        ]

    def sanitize(self, text: str) -> str:
        if not text:
            return text
            
        # Strip zero-width chars
        text = self.zero_width_pattern.sub('', text)
        
        # Strip HTML tags
        text = self.html_pattern.sub('', text)
        
        # We could also mask suspicious phrases, but since we aren't using an LLM at runtime,
        # they are harmless to our FAISS/Scorer. Still, it's good practice.
        for phrase in self.injection_phrases:
            # Case insensitive replace with [REDACTED]
            text = re.sub(phrase, '[REDACTED]', text, flags=re.IGNORECASE)
            
        return text.strip()
