import random
import pandas as pd
import re
from typing import Dict, Any, List, Callable, Tuple

# Load dataframe globally for random value generation
_df_cache = None

def _get_random_values_from_df(df: pd.DataFrame):
    """Generate random value functions based on actual dataframe values."""
    global _df_cache
    if _df_cache is None:
        # Filter out NaN and None values from serves column
        serves_unique = df['serves'].dropna().unique()
        serves_list = [s for s in serves_unique if s is not None and str(s).lower() != 'nan']
        
        _df_cache = {
            'calories_low': df['calories [cal]'].quantile([0.15, 0.25, 0.35, 0.45]).tolist(),
            'calories_high': df['calories [cal]'].quantile([0.55, 0.65, 0.75, 0.85]).tolist(),
            'calories_moderate_min': df['calories [cal]'].quantile([0.25, 0.30, 0.35]).tolist(),
            'calories_moderate_max': df['calories [cal]'].quantile([0.60, 0.65, 0.70, 0.75]).tolist(),
            'protein': df['protein [g]'].quantile([0.55, 0.65, 0.75, 0.85]).tolist(),
            'carbs': df['totalCarbohydrate [g]'].quantile([0.15, 0.25, 0.35, 0.45]).tolist(),
            'sodium': df['sodium [mg]'].quantile([0.35, 0.45, 0.55, 0.65]).tolist(),
            'duration': df['duration'].quantile([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]).tolist(),
            'rating': df['average_rating'].quantile([0.55, 0.65, 0.75, 0.85]).tolist(),
            'sugar': df['sugars [g]'].quantile([0.25, 0.35, 0.45, 0.55]).tolist(),
            'saturated_fat': df['saturatedFat [g]'].quantile([0.20, 0.30, 0.40]).tolist(),
            'serves': serves_list
        }
    return _df_cache

def _format_recipe_output(results, constraints):
    """Format recipe results into output string based on constraints."""
    output_parts = []
    for i, (_, row) in enumerate(results.iterrows()):
        parts = [f"{i+1}) {row['title']}"]
        
        # Add relevant nutritional info based on constraints
        if 'max_calories' in constraints or 'min_calories' in constraints:
            parts.append(f"{row['calories [cal]']:.1f} kcal")
        if 'max_protein' in constraints or 'min_protein' in constraints:
            parts.append(f"{row['protein [g]']:.1f} g protein")
        if 'max_sodium' in constraints or 'min_sodium' in constraints:
            parts.append(f"{row['sodium [mg]']:.1f} mg sodium")
        if 'max_carbs' in constraints or 'min_carbs' in constraints:
            parts.append(f"{row['totalCarbohydrate [g]']:.1f} g carbs")
        if 'max_sugar' in constraints or 'min_sugar' in constraints:
            parts.append(f"{row['sugars [g]']:.1f} g sugar")
        if 'max_fat' in constraints or 'min_fat' in constraints:
            parts.append(f"{row['totalFat [g]']:.1f} g fat")
        if 'max_duration' in constraints:
            parts.append(f"{int(row['duration'])} min")
        
        output_parts.append("—".join(parts))
    
    return "; ".join(output_parts)

def create_single_turn_example(instruction: str, df_filtered, parser, max_results: int = None) -> Dict[str, Any]:
    """Create a single-turn training example in instruct format."""
    constraints = parser.parse(instruction)
    
    # Pick N random recipes from the filtered results (needed to limit output size)
    if max_results:
        try:
            if len(df_filtered) <= max_results:
                results = df_filtered
            else:
                results = df_filtered.sample(n=max_results, replace=False)
        except Exception:
            results = df_filtered.head(max_results)
    else:
        results = df_filtered
    
    # Extract recipe IDs for evidence
    evidence_ids = results['recipe_id'].tolist()
    
    # Format output with recipe details
    output = _format_recipe_output(results, constraints)
    
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "constraints": constraints,
        "evidence_ids": evidence_ids
    }

# Map numbers to a list of possible wordings so we can randomize phrasing per example
NUM_TO_WORD = {
    1: ["one", "1", "a single", "single"],
    2: ["two", "2", "a couple of"],
    3: ["three", "3"],
    4: ["four", "4"],
    5: ["five", "5"]
}


def random_num_word(num_results: int) -> str:
    """Return a randomized wording for the requested number of results.

    Falls back to the numeric string if no variants are available.
    """
    variants = NUM_TO_WORD.get(num_results)
    if variants:
        return random.choice(variants)
    return str(num_results)


TEMPLATE_TAGS = {
    'diet': ['vegetarian', 'gluten-free', 'vegan', 'low-carb'],
    'meal': ['breakfast', 'lunch', 'dinner', 'dessert'],
    'misc': ['family-friendly', 'quick'],
    'macronutrient': ['protein', 'carbohydrate', 'fat']
}

# Helper functions to get random values from dataframe
def random_calorie_limit(df: pd.DataFrame, low=True):
    values = _get_random_values_from_df(df)
    return int(random.choice(values['calories_low'] if low else values['calories_high']))

def random_protein_min(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    return int(random.choice(values['protein']))

def random_carb_max(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    return int(random.choice(values['carbs']))

def random_sodium_max(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    return int(random.choice(values['sodium']))

def random_duration_max(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    return int(random.choice(values['duration']))

def random_rating_min(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    return round(random.choice(values['rating']), 1)

def random_serves(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    serve_val = random.choice(values['serves'])
    # Convert to regex pattern if needed
    if '-' in str(serve_val):
        parts = str(serve_val).split('-')
        if len(parts) == 2:
            return serve_val, f"{serve_val}|{parts[0]}|{parts[1]}"
    return serve_val, str(serve_val)

def random_sugar_max(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    return int(random.choice(values['sugar']))

def random_saturated_fat_max(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    return int(random.choice(values['saturated_fat']))

def random_calorie_moderate_range(df: pd.DataFrame):
    values = _get_random_values_from_df(df)
    cal_min = int(random.choice(values['calories_moderate_min']))
    cal_max = int(random.choice(values['calories_moderate_max']))
    return cal_min, cal_max

QueryFunc = Callable[[pd.DataFrame], pd.DataFrame]

# Single-turn templates
def single_template_1(df: pd.DataFrame, num_results: int) -> Tuple[str, QueryFunc]:
    num_word = random_num_word(num_results)
    tag_name = random.choice(TEMPLATE_TAGS['diet'])
    cal_limit = random_calorie_limit(df, low=True)
    sod_limit = random_sodium_max(df)
    dur_limit = random_duration_max(df)
    
    instruction = (f"Find {num_word} quick {tag_name} lunches under {cal_limit} kcal "
                   f"with less than {sod_limit} mg sodium and under {dur_limit} minutes.")
    
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_name, case=False, na=False)) &
        (df["calories [cal]"] < cal_limit) &
        (df["sodium [mg]"] < sod_limit) &
        (df["duration"] < dur_limit)
    ]
    return instruction, query_func

def single_template_2(df: pd.DataFrame, num_results: int) -> Tuple[str, QueryFunc]:
    num_word = random_num_word(num_results)
    tag_name = random.choice(TEMPLATE_TAGS['meal'][:3])  # breakfast, lunch, or dinner
    prot_limit = random_protein_min(df)
    dur_limit = random_duration_max(df)
    
    instruction = (f"Find {num_word} high-protein {tag_name}s over {prot_limit}g protein "
                   f"in under {dur_limit} minutes.")
    
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_name, case=False, na=False)) &
        (df["protein [g]"] > prot_limit) &
        (df["duration"] < dur_limit)
    ]
    return instruction, query_func

def single_template_3(df: pd.DataFrame, num_results: int) -> Tuple[str, QueryFunc]:
    num_word = random_num_word(num_results)
    tag_name = random.choice(['dinner', 'lunch'])
    carb_limit = random_carb_max(df)
    prot_limit = random_protein_min(df)
    
    instruction = (f"Find {num_word} low-carb {tag_name}s under {carb_limit}g total carbohydrates "
                   f"with at least {prot_limit}g protein.")
                   
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_name, case=False, na=False)) &
        (df["totalCarbohydrate [g]"] < carb_limit) &
        (df["protein [g]"] >= prot_limit)
    ]
    return instruction, query_func

def single_template_4(df: pd.DataFrame, num_results: int) -> Tuple[str, QueryFunc]:
    num_word = random_num_word(num_results)
    cal_limit = random_calorie_limit(df, low=True)
    sug_limit = random_sugar_max(df)
    fat_limit = random_saturated_fat_max(df)
    
    instruction = (f"Find {num_word} desserts under {cal_limit} kcal with less than "
                   f"{sug_limit}g sugar and low saturated fat (under {fat_limit}g).")
                   
    query_func = lambda df: df[
        (df["tags"].str.contains("dessert", case=False, na=False)) &
        (df["calories [cal]"] < cal_limit) &
        (df["sugars [g]"] < sug_limit) &
        (df["saturatedFat [g]"] < fat_limit)
    ]
    return instruction, query_func

def single_template_5(df: pd.DataFrame, num_results: int) -> Tuple[str, QueryFunc]:
    num_word = random_num_word(num_results)
    tag_name = random.choice(['gluten-free', 'vegetarian', 'vegan'])
    rating = random_rating_min(df)
    dur_limit = random_duration_max(df)
    
    instruction = (f"Find {num_word} highly-rated {tag_name} recipes with at least "
                   f"{rating} stars and under {dur_limit} minutes.")
                   
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_name, case=False, na=False)) &
        (df["average_rating"] >= rating) &
        (df["duration"] < dur_limit)
    ]
    return instruction, query_func

def single_template_6(df: pd.DataFrame, num_results: int) -> Tuple[str, QueryFunc]:
    num_word = random_num_word(num_results)
    tag_name = random.choice(['dinner', 'family-friendly'])
    serves_display, serves_regex = random_serves(df)
    cal_min, cal_max = random_calorie_moderate_range(df)
    
    instruction = (f"Find {num_word} {tag_name} recipes that serve {serves_display} people "
                   f"with moderate calories (between {cal_min} and {cal_max} kcal).")
                   
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_name, case=False, na=False)) &
        (df["serves"].str.contains(serves_regex, case=False, na=False, regex=True)) &
        (df["calories [cal]"] < cal_max) &
        (df["calories [cal]"] > cal_min)
    ]
    return instruction, query_func


def generate_random_examples(df: pd.DataFrame, parser: Any, num_examples: int, num_results_per_example: int) -> List[Dict[str, Any]]:
    """Generates a list of randomized training examples based on the single-turn templates."""
    
    # Single-turn templates
    template_generators = [
        single_template_1, single_template_2, single_template_3,
        single_template_4, single_template_5, single_template_6
    ]
    
    # <<< FIX: Create a list of valid numbers of results to ask for >>>
    # We'll use the keys from NUM_TO_WORD that are <= the requested max
    valid_nums = [k for k in NUM_TO_WORD.keys() if k <= num_results_per_example]
    # If the list is empty (e.g., user passed 0, or just a large number),
    # default to the full list of supported numbers.
    if not valid_nums:
        valid_nums = list(NUM_TO_WORD.keys())
        
    examples = []
    attempts = 0
    max_attempts = num_examples * 5
    used_recipe_sets = set()  # Track recipe combinations to prevent duplicates
    
    print(f"Attempting to generate {num_examples} examples...")
    
    while len(examples) < num_examples and attempts < max_attempts:
        # Pick template
        template_func = random.choice(template_generators)
        
        # <<< FIX: Pick a random number of results *for each example* >>>
        current_num_results = random.choice(valid_nums)
        
        instruction, query_func = template_func(df, current_num_results)
        results = query_func(df)
        
        # Create example if results found
        if not results.empty:
            # Format the answer
            example = create_single_turn_example(
                instruction=instruction,
                df_filtered=results,
                parser=parser,
                max_results=current_num_results # <<< Use the new random number
            )
            
            # Check for duplicate recipe sets (use sorted tuple for simpler comparison)
            recipe_set = tuple(sorted(example['evidence_ids']))
            if recipe_set not in used_recipe_sets:
                examples.append(example)
                used_recipe_sets.add(recipe_set)
            
        attempts += 1

    if len(examples) < num_examples:
        print(f"Failed to generate all the examples. Generated {len(examples)} out of {num_examples}.")
              
    print(f"Successfully generated {len(examples)} examples.")
    return examples




def create_multi_turn_example(conversation: List[str], df_filtered, parser, max_results: int = None) -> Dict[str, Any]:
    """Create a multi-turn training example in chat format."""
    # Parse conversation to extract constraints
    constraints = parser.parse_conversation(conversation)
    
    # Pick N random recipes from the filtered results (needed to limit output size)
    if max_results:
        try:
            if len(df_filtered) <= max_results:
                results = df_filtered
            else:
                results = df_filtered.sample(n=max_results, replace=False)
        except Exception:
            results = df_filtered.head(max_results)
    else:
        results = df_filtered
    
    # Extract recipe IDs for evidence
    evidence_ids = results['recipe_id'].tolist()
    
    # Format final assistant response
    final_output = _format_recipe_output(results, constraints)
    
    # Build messages in OpenAI chat format
    messages = []
    for i, msg in enumerate(conversation):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg})
    
    # Add final assistant response with recipes
    messages.append({"role": "assistant", "content": final_output})
    
    return {
        "messages": messages,
        "constraints": constraints,
        "evidence_ids": evidence_ids
    }


# Multi-turn templates
def multi_template_1(df: pd.DataFrame, num_results: int):
    """Quick diet lunch follow-up flow"""
    num_word = random_num_word(num_results)
    tag_name = random.choice(TEMPLATE_TAGS['diet'])
    cal_limit = random_calorie_limit(df, low=True)
    sod_limit = random_sodium_max(df)
    dur_limit = random_duration_max(df)

    conversation = [
        f"Show {tag_name} lunch options.",
        "Do you have any calorie or sodium preferences?",
        f"Under {cal_limit} kcal and less than {sod_limit} mg, please."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains(tag_name, case=False, na=False)) &
        (df["calories [cal]"] < cal_limit) &
        (df["sodium [mg]"] < sod_limit) &
        (df["duration"] < dur_limit)
    ]

    return conversation, query_func


def multi_template_2(df: pd.DataFrame, num_results: int):
    """Generic nutrient goal template"""
    num_word = random_num_word(num_results)
    tag_name = random.choice(TEMPLATE_TAGS['meal'][:3])
    
    # Choose random nutrient type and value
    nutrient_choices = [
        ('protein', random_protein_min(df), 'protein [g]', '>'),
        ('carbs', random_carb_max(df), 'totalCarbohydrate [g]', '<'),
    ]
    nutrient_name, nutrient_limit, nutrient_col, operator = random.choice(nutrient_choices)
    dur_limit = random_duration_max(df)

    conversation = [
        f"I need {tag_name} ideas.",
        f"What's your time constraint and {nutrient_name} goal?",
        f"Under {dur_limit} minutes, {'at least' if operator == '>' else 'under'} {nutrient_limit}g."
    ]

    if operator == '>':
        query_func = lambda df: df[
            (df["tags"].str.contains(tag_name, case=False, na=False)) &
            (df[nutrient_col] > nutrient_limit) &
            (df["duration"] < dur_limit)
        ]
    else:
        query_func = lambda df: df[
            (df["tags"].str.contains(tag_name, case=False, na=False)) &
            (df[nutrient_col] < nutrient_limit) &
            (df["duration"] < dur_limit)
        ]

    return conversation, query_func


def multi_template_3(df: pd.DataFrame, num_results: int):
    """Low-carb with protein goal"""
    num_word = random_num_word(num_results)
    tag_name = random.choice(['dinner', 'lunch'])
    carb_limit = random_carb_max(df)
    prot_limit = random_protein_min(df)

    conversation = [
        f"Find {tag_name} ideas.",
        "Are you looking for low-carb options?",
        f"Yes — under {carb_limit}g carbs and at least {prot_limit}g protein."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains(tag_name, case=False, na=False)) &
        (df["totalCarbohydrate [g]"] < carb_limit) &
        (df["protein [g]"] >= prot_limit)
    ]

    return conversation, query_func


def multi_template_4(df: pd.DataFrame, num_results: int):
    """Dessert with calorie constraint"""
    cal_limit = random_calorie_limit(df, low=True)
    sug_limit = random_sugar_max(df)
    fat_limit = random_saturated_fat_max(df)

    conversation = [
        "What desserts do you recommend?",
        "Are you looking for low-calorie or low-sugar?",
        f"Low-calorie, under {cal_limit} kcal, and low saturated fat (under {fat_limit}g)."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains("dessert", case=False, na=False)) &
        (df["calories [cal]"] < cal_limit) &
        (df["sugars [g]"] < sug_limit) &
        (df["saturatedFat [g]"] < fat_limit)
    ]

    return conversation, query_func


def multi_template_5(df: pd.DataFrame, num_results: int):
    """Quick low-carb chicken"""
    carb_limit = random_carb_max(df)
    dur_limit = random_duration_max(df)
    
    conversation = [
        "Show me chicken recipes.",
        "Would you prefer grilled, baked, or any particular style?",
        f"Something quick and low-carb, under {carb_limit}g carbs."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains("chicken", case=False, na=False)) &
        (df["totalCarbohydrate [g]"] < carb_limit) &
        (df["duration"] < dur_limit)
    ]

    return conversation, query_func


def multi_template_6(df: pd.DataFrame, num_results: int):
    """Low sodium vegetarian soup"""
    sod_limit = random_sodium_max(df)
    
    conversation = [
        "I want to make soup.",
        "Any dietary restrictions or sodium concerns?",
        f"Yes, low sodium under {sod_limit} mg and vegetarian."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains("soup", case=False, na=False)) &
        (df["tags"].str.contains("vegetarian", case=False, na=False)) &
        (df["sodium [mg]"] < sod_limit)
    ]

    return conversation, query_func



def generate_random_multi_turn_examples(df: pd.DataFrame, parser: Any, num_examples: int, num_results_per_example: int) -> List[Dict[str, Any]]:
    """Generate randomized multi-turn training examples using simple conversation templates."""
    
    # Multi-turn templates
    template_generators = [
        multi_template_1, multi_template_2, multi_template_3,
        multi_template_4, multi_template_5, multi_template_6
    ]
    
    examples: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = num_examples * 8
    used_recipe_sets = set()  # Track recipe combinations to prevent duplicates

    print(f"Attempting to generate {num_examples} multi-turn examples...")

    while len(examples) < num_examples and attempts < max_attempts:
        template_func = random.choice(template_generators)
        conversation, query_func = template_func(df, num_results_per_example)
        results = query_func(df)

        if not results.empty:
            example = create_multi_turn_example(
                conversation=conversation,
                df_filtered=results,
                parser=parser,
                max_results=num_results_per_example
            )
            
            # Check for duplicate recipe sets (use sorted tuple for simpler comparison)
            recipe_set = tuple(sorted(example['evidence_ids']))
            if recipe_set not in used_recipe_sets:
                examples.append(example)
                used_recipe_sets.add(recipe_set)
                
        attempts += 1

    if len(examples) < num_examples:
        print(f"Failed to generate all the multi-turn examples. Generated {len(examples)} out of {num_examples}.")

    print(f"Successfully generated {len(examples)} multi-turn examples.")
    return examples
