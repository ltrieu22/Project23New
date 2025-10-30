import re
import json
from typing import Dict, List, Set, Optional, Any
from nltk.corpus import wordnet as wn
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

class RecipeConstraintParser:
    def __init__(self):
        self.nutrients = ['protein', 'calorie', 'sugar', 'sodium', 'carbohydrate', 'carb', 'fat']
        self.time_keywords = ['minute', 'time', 'duration', 'hour', 'hr', 'min', 'minutes', 'mins', 'sec', 'seconds']
        self.diet_keywords = ['vegan', 'vegetarian', 'paleo', 'keto', 'gluten-free', 'dairy-free', 'low-carb']
        self.health_keywords = ['healthy', 'light', 'low-fat']
        self.number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        self.max_operators = ['no more than', 'less than', 'fewer than', 'under', 'below', 'maximum', 'max']
        self.min_operators = ['no less than', 'more than', 'at least', 'over', 'above', 'minimum', 'min', 'exceeding', 'exceed']
        
        self.units = ['g', 'gram', 'mg', 'milligram', 'kcal', 'calorie']
        
        self.food_roots = {
            'food', 'foodstuff', 'nutriment', 'dish', 'ingredient', 'edible',
            'poultry', 'meat', 'beef', 'pork', 'fish', 'seafood',
            'vegetable', 'veg', 'fruit', 'grain', 'cereal',
            'spice', 'herb', 'condiment', 'sauce',
            'beverage', 'drink', 'dairy', 'cheese', 'bread'
        }
        
        self.generic_nouns = {
            'recipe', 'recipes', 'option', 'options', 'idea', 'ideas', 'meal',
            'meals', 'food', 'dinner', 'dinners', 'lunch', 'lunches', 'breakfast',
            'breakfasts', 'dessert', 'desserts', 'appetizer', 'appetizers', 'soup',
            'soups', 'salad', 'salads', 'dish', 'dishes', 'serving', 'servings', 
            'person', 'people', 'g', 'mg', 'kcal', 'time', 'constraint', 
            'preference', 'preferences', 'style', 'diet', 'diets', 'sweet', 'snack',
            'snacks', 'appetiser', 'appetisers', 'starter', 'starters'
        }
        
        self.common_stop_words = set(['show', 'find', 'give', 'need', 'want', 'something', 'recipes', 'dishes', 
                                      'meals', 'options', 'ideas', 'make', 'breakfast', 'lunch', 'dinner', 
                                      'dessert', 'snack', 'quick', 'healthy', 'high', 'low', 'people', 'no'])
    
    def _is_synset_food_related(self, synset: wn.synset) -> bool:
        """Helper to check if a WordNet synset itself or its hypernyms are food-related."""
        
        synset_name = synset.name().split('.')[0]
        if synset_name in self.food_roots:
            return True
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            hypernym_name = hypernym.name().split('.')[0]
            if hypernym_name in self.food_roots:
                return True
            for h2 in hypernym.hypernyms():
                h2_name = h2.name().split('.')[0]
                if h2_name in self.food_roots:
                    return True
        return False

    def is_food_related(self, word: str) -> bool:
        synsets = wn.synsets(word, pos=wn.NOUN)
        for synset in synsets[:3]:  # Check first 3 senses
            if self._is_synset_food_related(synset):
                return True
        return False
    
    def get_ingredient_synonyms(self, ingredient: str) -> Set[str]:
        synonyms = set()
        synonyms.add(ingredient.lower())
        words = ingredient.lower().split()
        
        for word in words:
            synsets = wn.synsets(word, pos=wn.NOUN)
            for synset in synsets[:3]:  # Limit to top 3 synsets
                
                if self._is_synset_food_related(synset):
                    for lemma in synset.lemmas():
                        synonym = lemma.name().lower().replace('_', ' ')
                        if len(synonym.split()) <= 3:  # Avoid long compound terms
                            synonyms.add(synonym)
        
        compound = ingredient.lower().replace(' ', '_')
        synsets = wn.synsets(compound, pos=wn.NOUN)
        for synset in synsets[:3]:
            
            if self._is_synset_food_related(synset):
                for lemma in synset.lemmas():
                    synonym = lemma.name().lower().replace('_', ' ')
                    if len(synonym.split()) <= 3:
                        synonyms.add(synonym)
        
        return synonyms
    
    def extract_number(self, text: str) -> Optional[float]:
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        
        for word, num in self.number_words.items():
            if word in text.lower():
                return float(num)
        
        return None
    
    def parse_count(self, text: str) -> Optional[int]:
        patterns = [
            r'(?:find|show|give me)\s+(\w+)\s+(?:recipes|dishes|meals|options)',
            r'(\w+)\s+(?:vegan|vegetarian|keto|paleo)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                count_word = match.group(1)
                num = self.extract_number(count_word)
                if num:
                    return int(num)
        
        return None
    
    def parse_servings(self, text: str) -> Dict[str, int]:
        result = {}
        
        match = re.search(r'(?:serve|serving|around|about)?\s*(\d+)\s*[-–to]+\s*(\d+)\s*(?:people|servings?)?', text.lower())
        if match:
            result['min_servings'] = int(match.group(1))
            result['max_servings'] = int(match.group(2))
        
        return result
    
    def parse_nutrients(self, text: str, context_nutrients: Optional[List[str]] = None) -> Dict[str, float]:
        result = {}
        text_lower = text.lower()
        
        nutrient_map = {
            'calorie': 'calories', 'kcal': 'calories', 'calories': 'calories',
            'carb': 'carbs', 'carbohydrate': 'carbs', 'carbohydrates': 'carbs',
            'protein': 'protein', 'sugar': 'sugar', 'sodium': 'sodium',
            'fat': 'fat', 'saturated fat': 'saturated_fat'
        }
        
        # looking for "under 450 kcal", "at least 18g protein"
        for nutrient_key, nutrient_name in nutrient_map.items():
            for op in self.max_operators:
                op_pattern = re.escape(op)
                
                if op == 'less than':
                    pattern = rf'\b(?<!no\s){op_pattern}\s+(\d+(?:\.\d+)?)\s*(?:g|mg|kcal|gram|milligram|calorie)?\s*{nutrient_key}s?\b'
                else:
                    pattern = rf'\b{op_pattern}\s+(\d+(?:\.\d+)?)\s*(?:g|mg|kcal|gram|milligram|calorie)?\s*{nutrient_key}s?\b'

                match = re.search(pattern, text_lower)
                if match:
                    result[f'max_{nutrient_name}'] = float(match.group(1))
                    break
            
            for op in self.min_operators:
                op_pattern = re.escape(op)

                if op == 'more than':
                    pattern = rf'\b(?<!no\s){op_pattern}\s+(\d+(?:\.\d+)?)\s*(?:g|mg|kcal|gram|milligram|calorie)?\s*{nutrient_key}s?\b'
                else:
                    pattern = rf'\b{op_pattern}\s+(\d+(?:\.\d+)?)\s*(?:g|mg|kcal|gram|milligram|calorie)?\s*{nutrient_key}s?\b'
                
                match = re.search(pattern, text_lower)
                if match:
                    result[f'min_{nutrient_name}'] = float(match.group(1))
                    break
            
            if f'max_{nutrient_name}' not in result:
                for op in self.max_operators:
                    op_pattern = re.escape(op)
                    
                    if op == 'less than':
                        pattern = rf'{nutrient_key}s?\s+(?<!no\s){op_pattern}\s+(\d+(?:\.\d+)?)'
                    else:
                        pattern = rf'{nutrient_key}s?\s+{op_pattern}\s+(\d+(?:\.\d+)?)'

                    match = re.search(pattern, text_lower)
                    if match:
                        result[f'max_{nutrient_name}'] = float(match.group(1))
                        break
            
            if f'min_{nutrient_name}' not in result:
                for op in self.min_operators:
                    op_pattern = re.escape(op)

                    if op == 'more than':
                        pattern = rf'{nutrient_key}s?\s+(?<!no\s){op_pattern}\s+(\d+(?:\.\d+)?)'
                    else:
                        pattern = rf'{nutrient_key}s?\s+{op_pattern}\s+(\d+(?:\.\d+)?)'

                    match = re.search(pattern, text_lower)
                    if match:
                        result[f'min_{nutrient_name}'] = float(match.group(1))
                        break
        
        for nutrient_key, nutrient_name in nutrient_map.items():
            if f'max_{nutrient_name}' not in result:
                pattern = rf'<\s*(\d+(?:\.\d+)?)\s*(?:g|mg|kcal)?\s*{nutrient_key}s?\b'
                match = re.search(pattern, text_lower)
                if match:
                    result[f'max_{nutrient_name}'] = float(match.group(1))
            
            if f'min_{nutrient_name}' not in result:
                pattern = rf'>\s*(\d+(?:\.\d+)?)\s*(?:g|mg|kcal)?\s*{nutrient_key}s?\b'
                match = re.search(pattern, text_lower)
                if match:
                    result[f'min_{nutrient_name}'] = float(match.group(1))
        
        if context_nutrients:
            user_mentioned_nutrients = []
            for nutrient_key in nutrient_map.keys():
                if nutrient_key in text_lower:
                    user_mentioned_nutrients.append(nutrient_key)
            
            if not user_mentioned_nutrients:
                for context_nutrient in context_nutrients:
                    nutrient_name = nutrient_map.get(context_nutrient, context_nutrient)
                    
                    if f'max_{nutrient_name}' not in result:
                        for op in self.max_operators:
                            op_pattern = re.escape(op)
                            pattern = rf'\b{op_pattern}\s+(\d+(?:\.\d+)?)\s*(?:g|mg|kcal)\b'
                            match = re.search(pattern, text_lower)
                            if match:
                                result[f'max_{nutrient_name}'] = float(match.group(1))
                                break
                    
                    if f'min_{nutrient_name}' not in result:
                        for op in self.min_operators:
                            op_pattern = re.escape(op)
                            pattern = rf'\b{op_pattern}\s+(\d+(?:\.\d+)?)\s*(?:g|mg|kcal)\b'
                            match = re.search(pattern, text_lower)
                            if match:
                                result[f'min_{nutrient_name}'] = float(match.group(1))
                                break
        
        return result
    
    def parse_time(self, text: str) -> Dict[str, float]:
        result = {}
        text_lower = text.lower()
        
        if 'quick' in text_lower:
            result['max_duration'] = 30.0
        
        patterns = [
            rf'(?:' + '|'.join(self.max_operators) + r')\s+(\d+)\s*(?:min|minute|minutes)',
            rf'in\s+(?:under|about)?\s*(\d+)\s*(?:min|minute|minutes)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                result['max_duration'] = float(match.group(1))
                break
        
        return result
    
    def parse_diet(self, text: str) -> List[str]:
        diets = []
        text_lower = text.lower()
        
        for diet in self.diet_keywords:
            if diet in text_lower:
                diets.append(diet)
        
        return diets
    
    def parse_health_category(self, text: str) -> List[str]:
        categories = []
        text_lower = text.lower()
        
        if 'healthy' in text_lower:
            categories.extend(['healthy-2', 'healthy'])
        
        for keyword in self.health_keywords:
            if keyword in text_lower and keyword not in categories:
                categories.append(keyword)
        
        return categories
    
    def parse_ingredients(self, text: str) -> Dict[str, List[str]]:
        result = {}
        text_lower = text.lower()
        
        stop_phrases = set(['no more than', 'no less than', 'more than', 'less than'])
        nutrient_words = set(self.nutrients + ['calories', 'carbs'])
        
        include_patterns = [
            r'with\s+([\w\s]+?)(?=\s*(?:,|and\b|under\b|below\b|over\b|above\b|less\b|more\b|\.|$))',
            r'containing\s+([\w\s]+?)(?=\s*(?:,|and\b|without\b|\.|$))',
            r'(?:include|using)\s+([\w\s]+?)(?=\s*(?:,|and\b|\.|$))',
        ]
        
        exclude_patterns = [
            r'without\s+([\w\s]+?)(?=\s*(?:,|and\b|\.|$))',
            r'exclude\s+([\w\s]+?)(?=\s*(?:,|and\b|\.|$))',
        ]
        
        included = set()
        excluded = set()
        
        for pattern in include_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                ingredient = match.group(1).strip()
                
                if ingredient in self.common_stop_words:
                    continue
                
                if any(phrase in ingredient for phrase in stop_phrases):
                    continue
                if any(nutrient in ingredient for nutrient in nutrient_words):
                    continue
                
                ingredient = re.sub(r'\s+(under|below|over|above|less|more|than|at|least).*', '', ingredient)
                ingredient = ingredient.strip()
                
                if ingredient and len(ingredient.split()) <= 3:
                    synonyms = self.get_ingredient_synonyms(ingredient)
                    included.update(synonyms)
        
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)  # Words with 3+ chars
        for word in words:
            # Skip common words, nutrients, diet keywords, etc.
            if word in self.common_stop_words:
                continue
            if word in nutrient_words:
                continue
            if word in self.diet_keywords:
                continue
            if word in self.health_keywords:
                continue
            if word in self.time_keywords:
                continue
            if word in self.generic_nouns:  # Skip generic food nouns
                continue
            
            if self.is_food_related(word):
                synonyms = self.get_ingredient_synonyms(word)
                food_synonyms = {
                    s for s in synonyms 
                    if s not in self.generic_nouns  # Filter out generic nouns from synonyms
                }
                included.update(food_synonyms)
        
        
        matches = re.finditer(exclude_patterns[0], text_lower)
        for match in matches:
            ingredient = match.group(1).strip()
            
            if ingredient in self.common_stop_words:
                continue
            
            if any(phrase in ingredient for phrase in stop_phrases):
                continue
            if any(nutrient in ingredient for nutrient in nutrient_words):
                continue
            
            ingredient = re.sub(r'\s+(under|below|over|above|less|more|than|at|least).*', '', ingredient)
            ingredient = ingredient.strip()
            
            if ingredient and len(ingredient.split()) <= 3:
                synonyms = self.get_ingredient_synonyms(ingredient)
                excluded.update(synonyms)
        
        no_pattern = r'\bno\s+([\w]+)(?=\s*(?:,|and\s+no|\.|$))'
        matches = re.finditer(no_pattern, text_lower)
        for match in matches:
            ingredient = match.group(1).strip()
            
            if ingredient in ['more', 'less', 'fewer']:
                continue
            if ingredient in nutrient_words:
                continue
            
            if ingredient and len(ingredient) > 2:  # At least 3 characters
                synonyms = self.get_ingredient_synonyms(ingredient)
                excluded.update(synonyms)
        
        matches = re.finditer(exclude_patterns[1], text_lower)
        for match in matches:
            ingredient = match.group(1).strip()
            
            if ingredient in self.common_stop_words:
                continue
            
            if any(phrase in ingredient for phrase in stop_phrases):
                continue
            if any(nutrient in ingredient for nutrient in nutrient_words):
                continue
            
            ingredient = re.sub(r'\s+(under|below|over|above|less|more|than|at|least).*', '', ingredient)
            ingredient = ingredient.strip()
            
            if ingredient and len(ingredient.split()) <= 3:
                synonyms = self.get_ingredient_synonyms(ingredient)
                excluded.update(synonyms)
        
        if included and excluded:
            included = included - excluded
        
        if included:
            result['include_ingredients'] = sorted(list(included))
        
        if excluded:
            result['exclude_ingredients'] = sorted(list(excluded))
        
        return result
    
    def parse(self, query: str) -> Dict[str, Any]:
        """Parse a single query into constraints."""
        constraints = {}
        
        # nutrients first to avoid ingredient false positives
        count = self.parse_count(query)
        if count:
            constraints['count'] = count
        
        servings = self.parse_servings(query)
        constraints.update(servings)
        
        # BEFORE ingredients to avoid capturing operators
        nutrients = self.parse_nutrients(query)
        constraints.update(nutrients)
        
        time_constraints = self.parse_time(query)
        constraints.update(time_constraints)
        
        diets = self.parse_diet(query)
        if diets:
            constraints['diet'] = diets
        
        health = self.parse_health_category(query)
        if health:
            constraints['health_category'] = health
        
        ingredients = self.parse_ingredients(query)
        constraints.update(ingredients)
        
        return constraints
    
    def parse_with_context(self, user_message: str, context_hints: List[str]) -> Dict[str, Any]:
        constraints = {}
        context_lower = ' '.join(context_hints).lower()
        
        context_nutrients = []
        nutrient_map = {
            'calorie': 'calorie', 'calories': 'calorie', 'kcal': 'calorie',
            'carb': 'carb', 'carbohydrate': 'carb', 'carbohydrates': 'carb',
            'protein': 'protein', 'sugar': 'sugar', 'sodium': 'sodium',
            'fat': 'fat'
        }
        
        for nutrient_word, canonical in nutrient_map.items():
            if nutrient_word in context_lower:
                if canonical not in context_nutrients:
                    context_nutrients.append(canonical)
        
        # Parse with context awareness
        if context_nutrients:
            # Parse nutrients with context
            nutrients = self.parse_nutrients(user_message, context_nutrients)
            constraints.update(nutrients)
        else:
            # Normal parsing
            nutrients = self.parse_nutrients(user_message)
            constraints.update(nutrients)
        
        count = self.parse_count(user_message)
        if count:
            constraints['count'] = count
        
        servings = self.parse_servings(user_message)
        constraints.update(servings)
        
        time_constraints = self.parse_time(user_message)
        constraints.update(time_constraints)
        
        diets = self.parse_diet(user_message)
        if diets:
            constraints['diet'] = diets
        
        health = self.parse_health_category(user_message)
        if health:
            constraints['health_category'] = health
        
        ingredients = self.parse_ingredients(user_message)
        constraints.update(ingredients)
        
        return constraints
    
    def parse_conversation(self, messages: List[str]) -> Dict[str, Any]:
        """Parse multi-turn conversation, preserving ingredients across turns."""
        all_constraints = {}
        
        all_include_ingredients = set()
        all_exclude_ingredients = set()
        all_diets = set()
        all_health = set()

        for i, message in enumerate(messages):
            # Determine if this is user or assistant message
            is_user = (i % 2 == 0)
            
            if is_user:
                # Check if previous message (assistant) provides context hints
                context_hints = []
                if i > 0:
                    assistant_msg = messages[i - 1]
                    context_hints.append(assistant_msg)
                
                # Parse with context
                if context_hints:
                    constraints = self.parse_with_context(message, context_hints)
                else:
                    constraints = self.parse(message)
                
                # Merge constraints
                for key, value in constraints.items():
                    if key == 'include_ingredients':
                        all_include_ingredients.update(value)
                    elif key == 'exclude_ingredients':
                        all_exclude_ingredients.update(value)
                    elif key == 'diet':
                        all_diets.update(value)
                    elif key == 'health_category':
                        all_health.update(value)
                    else:
                        # For non-list, non-additive values, update with latest
                        all_constraints[key] = value
        
        if all_include_ingredients:
            # Ensure excluded items are not in the final include list
            all_include_ingredients = all_include_ingredients - all_exclude_ingredients
            all_constraints['include_ingredients'] = sorted(list(all_include_ingredients))
        if all_exclude_ingredients:
            all_constraints['exclude_ingredients'] = sorted(list(all_exclude_ingredients))
        if all_diets:
            all_constraints['diet'] = sorted(list(all_diets))
        if all_health:
            all_constraints['health_category'] = sorted(list(all_health))

        return all_constraints

parser = RecipeConstraintParser()