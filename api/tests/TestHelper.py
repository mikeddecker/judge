class TestHelper:
    def generate_empty_strings():
        return [ None, "", " ", "\n", "\r", "  ", "\t ", "\t", " \r\n " ]
    
    def generate_invalid_ids():
        return [-123_456, -5, -1, 0, None, "string"]
    
    def generate_invalid_strings_only_word_digit_underscore():
        return ["hello!", "dot.dot.dot", "dotted.name", "seme%", "0623()", "§dsqk"]
    
    def generate_invalid_strings_only_word_digit_underscore_extensions():
        return ["hello!", "dot.dot.dot", "seme%", "0623()", "§dsqk"]
    
    def generate_zero_to_one_included(valid=True):
        return [0.0, 0.001, 0.44, 0.999, 1.0] if valid else [-1.5, -0.001, 1.001, 1.00000001]
