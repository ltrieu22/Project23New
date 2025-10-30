import random
import pandas as pd
from typing import Dict, Any, List, Callable, Tuple

def create_single_turn_example(instruction: str, df_filtered, parser, max_results: int = None) -> Dict[str, Any]:
    """Create a single-turn training example in instruct format."""
    constraints = parser.parse(instruction)
    
    # Pick a random subset when limiting results
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
    
    output = "; ".join(output_parts)
    
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "constraints": constraints,
        "evidence_ids": evidence_ids
    }


NUM_TO_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

TAG_POOLS = {
    'diet': [('vegetarian', 'vegetarian'), ('gluten-free', 'gluten-free'), ('vegan', 'vegan'), ('low-carb', 'low-carb')],
    'meal': [('breakfast', 'breakfast'), ('lunch', 'lunch'), ('dinner', 'dinner'), ('dessert', 'dessert')],
    'misc': [('family-friendly', 'family-friendly'), ('quick', 'quick')],
    'macronutrient': [('protein', 'protein'), ('carbohydrate', 'carbohydrate'), ('fat', 'fat')]
}

def random_calorie_limit(low=True):
    return random.choice([250, 300, 350, 400]) if low else random.choice([500, 600, 700, 800, 900])

def random_protein_min():
    return random.choice([15, 20, 25, 30])

def random_carb_max():
    return random.choice([10, 15, 20, 30, 40, 50, 60])

def random_sodium_max():
    return random.choice([300, 400, 500, 600, 700])

def random_duration_max():
    return random.choice([15, 20, 30, 45, 60, 75, 90, 120])

def random_rating_min():
    return random.choice([3.0, 3.5, 4.0, 4.5])

def random_serves():
    return random.choice([('4-6', '4-6|4|6'), ('6-8', '6-8|6|8'), ('8-10', '8-10|8|10'), ('10-12', '10-12|10|12')]) 

QueryFunc = Callable[[pd.DataFrame], pd.DataFrame]
# Templates
def _template_1(num_results: int) -> Tuple[str, QueryFunc]:
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    tag_val, tag_name = random.choice(TAG_POOLS['diet'])
    cal_limit = random_calorie_limit(low=True)
    sod_limit = random_sodium_max()
    dur_limit = random_duration_max()
    
    instruction = (f"Find {num_word} quick {tag_name} lunches under {cal_limit} kcal "
                   f"with less than {sod_limit} mg sodium and under {dur_limit} minutes.")
    
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_val, case=False, na=False)) &
        (df["calories [cal]"] < cal_limit) &
        (df["sodium [mg]"] < sod_limit) &
        (df["duration"] < dur_limit)
    ]
    return instruction, query_func

def _template_2(num_results: int) -> Tuple[str, QueryFunc]:
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    tag_val, tag_name = random.choice(TAG_POOLS['meal'][:3]) # breakfast, lunch, or dinner
    prot_limit = random_protein_min()
    dur_limit = random_duration_max()
    
    instruction = (f"Find {num_word} high-protein {tag_name}s over {prot_limit}g protein "
                   f"in under {dur_limit} minutes.")
    
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_val, case=False, na=False)) &
        (df["protein [g]"] > prot_limit) &
        (df["duration"] < dur_limit)
    ]
    return instruction, query_func

def _template_3(num_results: int) -> Tuple[str, QueryFunc]:
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    tag_val, tag_name = random.choice([('dinner', 'dinner'), ('lunch', 'lunch')])
    carb_limit = random_carb_max()
    prot_limit = random_protein_min()
    
    instruction = (f"Find {num_word} low-carb {tag_name}s under {carb_limit}g total carbohydrates "
                   f"with at least {prot_limit}g protein.")
                   
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_val, case=False, na=False)) &
        (df["totalCarbohydrate [g]"] < carb_limit) &
        (df["protein [g]"] >= prot_limit)
    ]
    return instruction, query_func

def _template_4(num_results: int) -> Tuple[str, QueryFunc]:
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    cal_limit = random_calorie_limit(low=True)
    sug_limit = random.choice([10, 15, 20, 25])
    fat_limit = random.choice([3, 5, 8]) # Saturated fat limits
    
    instruction = (f"Find {num_word} desserts under {cal_limit} kcal with less than "
                   f"{sug_limit}g sugar and low saturated fat (under {fat_limit}g).")
                   
    query_func = lambda df: df[
        (df["tags"].str.contains("dessert", case=False, na=False)) &
        (df["calories [cal]"] < cal_limit) &
        (df["sugars [g]"] < sug_limit) &
        (df["saturatedFat [g]"] < fat_limit)
    ]
    return instruction, query_func

def _template_5(num_results: int) -> Tuple[str, QueryFunc]:
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    tag_val, tag_name = random.choice([('gluten-free', 'gluten-free'), ('vegetarian', 'vegetarian'), ('vegan', 'vegan')])
    rating = random_rating_min()
    dur_limit = random_duration_max()
    
    instruction = (f"Find {num_word} highly-rated {tag_name} recipes with at least "
                   f"{rating} stars and under {dur_limit} minutes.")
                   
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_val, case=False, na=False)) &
        (df["average_rating"] >= rating) &
        (df["duration"] < dur_limit)
    ]
    return instruction, query_func

def _template_6(num_results: int) -> Tuple[str, QueryFunc]:
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    tag_val, tag_name = random.choice([('dinner', 'dinner'), ('family-friendly', 'family-friendly')])
    serves_display, serves_regex = random_serves()
    cal_min = random.randint(250, 350)
    cal_max = random.randint(500, 650)
    
    instruction = (f"Find {num_word} {tag_name} recipes that serve {serves_display} people "
                   f"with moderate calories (between {cal_min} and {cal_max} kcal).")
                   
    query_func = lambda df: df[
        (df["tags"].str.contains(tag_val, case=False, na=False)) &
        (df["serves"].str.contains(serves_regex, case=False, na=False, regex=True)) &
        (df["calories [cal]"] < cal_max) &
        (df["calories [cal]"] > cal_min)
    ]
    return instruction, query_func

def generate_random_examples(df: pd.DataFrame, parser: Any, num_examples: int, num_results_per_example: int) -> List[Dict[str, Any]]:
    """
    Generates a list of randomized training examples based on the 6 templates.
    """
    
    # Our templates
    template_generators = [
        _template_1, _template_2, _template_3,
        _template_4, _template_5, _template_6
    ]
    
    examples = []
    attempts = 0
    max_attempts = num_examples * 5
    
    print(f"Attempting to generate {num_examples} examples...")
    
    while len(examples) < num_examples and attempts < max_attempts:
        # Pick template
        template_func = random.choice(template_generators)
        
        instruction, query_func = template_func(num_results_per_example)
        results = query_func(df)
        
        # Create example if results found
        if not results.empty:
            # Format the answer
            example = create_single_turn_example(
                instruction=instruction,
                df_filtered=results,
                parser=parser,
                max_results=num_results_per_example
            )
            examples.append(example)
        attempts += 1

    if len(examples) < num_examples:
        print(f"Failed to generate all the examples. Generated {len(examples)} out of {num_examples}.")
              
    print(f"Successfully generated {len(examples)} examples.")
    return examples



def create_multi_turn_example(conversation: List[str], df_filtered, parser, max_results: int = None) -> Dict[str, Any]:
    """Create a multi-turn training example in chat format."""
    # Parse conversation to extract constraints
    constraints = parser.parse_conversation(conversation)
    
    # When limiting results, pick a random subset rather than always taking the top rows.
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
        if 'max_duration' in constraints:
            parts.append(f"{int(row['duration'])} min")
        
        output_parts.append("—".join(parts))
    
    final_output = "; ".join(output_parts)
    
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

# --- New: randomized multi-turn generator (parallels single-turn generator) ---
import random

def _mt_template_1(num_results: int):
    # Quick diet lunch follow-up flow
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    tag_val, tag_name = random.choice(TAG_POOLS['diet'])
    cal_limit = random_calorie_limit(low=True)
    sod_limit = random_sodium_max()
    dur_limit = random_duration_max()

    conversation = [
        f"Show {tag_name} lunch options.",
        "Do you have any calorie or sodium preferences?",
        f"Under {cal_limit} kcal and less than {sod_limit} mg, please."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains(tag_val, case=False, na=False)) &
        (df["calories [cal]"] < cal_limit) &
        (df["sodium [mg]"] < sod_limit) &
        (df["duration"] < dur_limit)
    ]

    return conversation, query_func


def _mt_template_2(num_results: int):
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    tag_val, tag_name = random.choice(TAG_POOLS['meal'][:3])
    prot_limit = random_protein_min()
    dur_limit = random_duration_max()

    conversation = [
        f"I need {tag_name} ideas.",
        "What's your time constraint and {nutrient} goal?",
        f"Under {dur_limit} minutes, at least {prot_limit}g."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains(tag_val, case=False, na=False)) &
        (df["protein [g]"] > prot_limit) &
        (df["duration"] < dur_limit)
    ]

    return conversation, query_func


def _mt_template_3(num_results: int):
    num_word = NUM_TO_WORD.get(num_results, str(num_results))
    tag_val, tag_name = random.choice([('dinner', 'dinner'), ('lunch', 'lunch')])
    carb_limit = random_carb_max()
    prot_limit = random_protein_min()

    conversation = [
        f"Find {tag_name} ideas.",
        "Are you looking for low-carb options?",
        f"Yes — under {carb_limit}g carbs and at least {prot_limit}g protein."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains(tag_val, case=False, na=False)) &
        (df["totalCarbohydrate [g]"] < carb_limit) &
        (df["protein [g]"] >= prot_limit)
    ]

    return conversation, query_func


def _mt_template_4(num_results: int):
    cal_limit = random_calorie_limit(low=True)
    sug_limit = random.choice([10, 15, 20, 25])
    fat_limit = random.choice([3, 5, 8])

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


def _mt_template_5(num_results: int):
    conversation = [
        "Show me chicken recipes.",
        "Would you prefer grilled, baked, or any particular style?",
        "Something quick and low-carb, under 20g carbs."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains("chicken", case=False, na=False)) &
        (df["totalCarbohydrate [g]"] < 20) &
        (df["duration"] < 30)
    ]

    return conversation, query_func


def _mt_template_6(num_results: int):
    conversation = [
        "I want to make soup.",
        "Any dietary restrictions or sodium concerns?",
        "Yes, low sodium under 400 mg and vegetarian."
    ]

    query_func = lambda df: df[
        (df["tags"].str.contains("soup", case=False, na=False)) &
        (df["tags"].str.contains("vegetarian", case=False, na=False)) &
        (df["sodium [mg]"] < 400)
    ]

    return conversation, query_func


MT_TEMPLATES = [_mt_template_1, _mt_template_2, _mt_template_3, _mt_template_4, _mt_template_5, _mt_template_6]


def generate_random_multi_turn_examples(df: Any, parser: Any, num_examples: int, num_results_per_example: int) -> List[Dict[str, Any]]:
    """Generate randomized multi-turn training examples using simple conversation templates."""
    examples: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = num_examples * 8

    print(f"Attempting to generate {num_examples} multi-turn examples...")

    while len(examples) < num_examples and attempts < max_attempts:
        template_func = random.choice(MT_TEMPLATES)
        conversation, query_func = template_func(num_results_per_example)
        results = query_func(df)

        if not results.empty:
            example = create_multi_turn_example(conversation=conversation, df_filtered=results, parser=parser, max_results=num_results_per_example)
            examples.append(example)
        attempts += 1

    if len(examples) < num_examples:
        print(f"Failed to generate all the multi-turn examples. Generated {len(examples)} out of {num_examples}.")

    print(f"Successfully generated {len(examples)} multi-turn examples.")
    return examples