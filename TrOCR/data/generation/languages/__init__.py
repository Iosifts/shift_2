import os
import importlib

class LanguagePatterns:
    """Base class for language-specific text patterns"""
    
    def __init__(self):
        self.patterns = {
            'titles': [],
            'honorifics': [],
            'positions': [],
            'dates': [],
            'numbers': [],
            'locations': [],
            'signatures': [],
            'currency': [],
            'common_abbrev': []
        }
        
        # Initialize with language-specific patterns
        self.init_patterns()
    
    def init_patterns(self):
        """Override this method to define language-specific patterns"""
        pass
    
    def get_patterns(self):
        """Returns all patterns for the language"""
        return self.patterns

def get_available_languages():
    """
    Discovers available language modules in the languages directory.
    Returns a list of valid language codes.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    languages = []
    
    for file in os.listdir(current_dir):
        if file.endswith('.py') and not file.startswith('__'):
            lang_code = file[:-3]  # Remove .py extension
            languages.append(lang_code)
    
    return languages

def load_language_patterns(language_code):
    """
    Dynamically loads language patterns for the specified language code.
    Returns None if language not found.
    """
    try:
        module = importlib.import_module(f'.{language_code}', package='data.generation.languages')
        pattern_class = getattr(module, f"{language_code.capitalize()}Patterns")
        return pattern_class().get_patterns()
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not load patterns for language '{language_code}': {e}")
        return None
