import unittest
from recipe_parser import RecipeConstraintParser


class TestRecipeParserSingleTurn(unittest.TestCase):
    """Test cases for single-turn recipe constraint parsing."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize parser once for all tests."""
        cls.parser = RecipeConstraintParser()
    
    def _check_constraints(self, result, expected):
        """Helper method to verify parsed constraints match expected values."""
        for key, expected_value in expected.items():
            self.assertIn(key, result, f"Missing key: {key}")
            
            if isinstance(expected_value, list):
                actual_value = result.get(key, [])
                for item in expected_value:
                    self.assertIn(item, actual_value, 
                                f"Expected {item} in {key}, got {actual_value}")
            elif isinstance(expected_value, (int, float)):
                actual_value = result.get(key, 0)
                self.assertAlmostEqual(actual_value, expected_value, places=2,
                                     msg=f"{key}: expected {expected_value}, got {actual_value}")
            else:
                self.assertEqual(result.get(key), expected_value,
                               f"{key}: expected {expected_value}, got {result.get(key)}")
    
    def test_vegan_dinner_with_calories_and_protein(self):
        input_text = "Find two vegan dinners under 450 kcal with at least 18 g protein."
        expected = {
            "count": 2,
            "diet": ["vegan"],
            "max_calories": 450.0,
            "min_protein": 18.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_breakfast_with_protein_sugar_time(self):
        input_text = "I need breakfast with protein over 20g, sugar under 10g, in 15 minutes."
        expected = {
            "min_protein": 20.0,
            "max_sugar": 10.0,
            "max_duration": 15.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_low_carb_with_protein(self):
        input_text = "Give me low-carb meals under 30g carbohydrates with protein exceeding 20g."
        expected = {
            "diet": ["low-carb"],
            "max_carbs": 30.0,
            "min_protein": 20.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_include_garbanzo_beans(self):
        input_text = "Find recipes with garbanzo beans"
        expected = {
            "include_ingredients": ["chickpea", "garbanzo"]
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_chickpeas_without_peanuts(self):
        input_text = "Show dishes containing chickpeas without peanuts"
        expected = {
            "include_ingredients": ["chickpea", "garbanzo"],
            "exclude_ingredients": ['arachis hypogaea', 'peanut', 'peanut vine', 'peanuts']
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_protein_and_calories(self):
        input_text = "Find meals with protein exceeding 30g and calories below 500"
        expected = {
            "min_protein": 30.0,
            "max_calories": 500.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_protein_but_under_calories(self):
        input_text = "I want something with at least 15g protein but under 300 calories"
        expected = {
            "min_protein": 15.0,
            "max_calories": 300.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_vegan_carbs_protein(self):
        input_text = "Find vegan options with no more than 25g carbs and at least 10g protein"
        expected = {
            "diet": ["vegan"],
            "max_carbs": 25.0,
            "min_protein": 10.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_vegetarian_lunches_with_sodium(self):
        input_text = "Show me 3 vegetarian lunches under 400 kcal with less than 600 mg sodium."
        expected = {
            "count": 3,
            "diet": ["vegetarian"],
            "max_calories": 400.0,
            "max_sodium": 600.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_high_protein_quick_breakfast(self):
        input_text = "I want high-protein breakfasts over 25g protein in under 20 minutes."
        expected = {
            "min_protein": 25.0,
            "max_duration": 20.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_desserts_calories_sugar(self):
        input_text = "Find desserts under 300 kcal with less than 20g sugar and low saturated fat."
        expected = {
            "max_calories": 300.0,
            "max_sugar": 20.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_dinners_with_servings_range(self):
        input_text = "Find dinners that serve 6-8 people with moderate calories."
        expected = {
            "min_servings": 6,
            "max_servings": 8
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_healthy_soup_without_butter(self):
        input_text = "Find healthy soup recipes without butter, sodium less than 400mg."
        expected = {
            "health_category": ["healthy-2", "healthy"],
            "exclude_ingredients": ["butter"],
            "max_sodium": 400.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_quick_pasta_chickpeas_vegetarian(self):
        input_text = "Show quick pasta options with chickpeas, under 450 kcal, vegetarian."
        expected = {
            "include_ingredients": ["chickpea", "garbanzo"],
            "max_calories": 450.0,
            "diet": ["vegetarian"]
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)
    
    def test_scallions_no_peanuts(self):
        input_text = "I want meals with scallions and no peanuts, under 500 calories."
        expected = {
            "include_ingredients": ["scallion", "green onion", "spring onion"],
            "exclude_ingredients": ['arachis hypogaea', 'peanut', 'peanut vine', 'peanuts'],
            "max_calories": 500.0
        }
        result = self.parser.parse(input_text)
        self._check_constraints(result, expected)


class TestRecipeParserMultiTurn(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.parser = RecipeConstraintParser()
    
    def _check_constraints(self, result, expected):
        for key, expected_value in expected.items():
            self.assertIn(key, result, f"Missing key: {key}")
            
            if isinstance(expected_value, list):
                actual_value = result.get(key, [])
                for item in expected_value:
                    self.assertIn(item, actual_value, 
                                f"Expected {item} in {key}, got {actual_value}")
            elif isinstance(expected_value, (int, float)):
                actual_value = result.get(key, 0)
                self.assertAlmostEqual(actual_value, expected_value, places=2,
                                     msg=f"{key}: expected {expected_value}, got {actual_value}")
            else:
                self.assertEqual(result.get(key), expected_value,
                               f"{key}: expected {expected_value}, got {result.get(key)}")
    
    def test_pasta_with_calorie_diet_preferences(self):
        conversation = [
            "Show quick pasta options.",
            "Do you have calorie or diet preferences?",
            "<450 kcal, vegetarian.",
        ]
        expected = {
            "max_calories": 450.0,
            "diet": ["vegetarian"]
        }
        result = self.parser.parse_conversation(conversation)
        self._check_constraints(result, expected)
    
    def test_breakfast_with_time_and_protein(self):
        conversation = [
            "I need breakfast ideas.",
            "What's your time constraint and protein goal?",
            "Under 15 minutes, at least 20g.",
        ]
        expected = {
            "max_duration": 15.0,
            "min_protein": 20.0
        }
        result = self.parser.parse_conversation(conversation)
        self._check_constraints(result, expected)
    
    def test_desserts_with_calorie_preference(self):
        conversation = [
            "What desserts do you recommend?",
            "Are you looking for something low-calorie or low-sugar?",
            "Low-calorie, under 200 kcal.",
        ]
        expected = {
            "max_calories": 200.0
        }
        result = self.parser.parse_conversation(conversation)
        self._check_constraints(result, expected)
    
    def test_chicken_with_style_preference(self):
        conversation = [
            "Show me chicken recipes.",
            "Would you prefer grilled, baked, or any specific style?",
            "Something quick and low-carb, under 20g carbs.",
        ]
        expected = {
            "max_duration": 30.0,
            "diet": ["low-carb"],
            "max_carbs": 20.0
        }
        result = self.parser.parse_conversation(conversation)
        self._check_constraints(result, expected)
    
    def test_soup_with_dietary_restrictions(self):
        conversation = [
            "I want to make soup.",
            "Any dietary restrictions or sodium concerns?",
            "Yes, low sodium under 400mg and vegetarian.",
        ]
        expected = {
            "max_sodium": 400.0,
            "diet": ["vegetarian"]
        }
        result = self.parser.parse_conversation(conversation)
        self._check_constraints(result, expected)
    
    def test_party_appetizer_with_servings(self):
        conversation = [
            "I need a party appetizer.",
            "How many people are you serving?",
            "Around 10-12 people.",
        ]
        expected = {
            "min_servings": 10,
            "max_servings": 12
        }
        result = self.parser.parse_conversation(conversation)
        self._check_constraints(result, expected)


if __name__ == '__main__':
    unittest.main()