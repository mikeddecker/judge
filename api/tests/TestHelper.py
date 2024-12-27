class TestHelper:
    def generate_empty_strings():
        return [ None, "", " ", "\n", "\r", "  ", "\t ", "\t", " \r\n " ]
    
    def generate_invalid_ids():
        return [-123_456, -5, -1, 0, None, "string"]
    
    def generate_invalid_strings_only_word_digit_underscore():
        return ["hello!", "dot.dot.dot", "dotted.name", "seme%", "0623()", "§dsqk"]
    
    def generate_invalid_strings_only_word_digit_underscore_extensions():
        return ["hello!", "dot.dot.dot", "seme%", "0623()", "§dsqk"]
