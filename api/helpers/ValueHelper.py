import re

class ValueHelper:
    def check_raise_id(id: int):
        if not isinstance(id, int):
            raise ValueError(f"Id must be of type {int}, got {id}")
        if id <= 0:
            raise ValueError(f"Id must be strict positive integer, got {id}")
        
    def check_raise_string(val: str):
        if val is None:
            raise ValueError(f"String may not be none")
        if val.isspace() or val == "":
            raise ValueError(f"String may not be empty")
    
    def check_raise_string_only_abc123(val: str):
        """Checks the strings to only allow examples below
        words
        digits
        2024
        word_digits_and_underscores_1999
        """
        ValueHelper.check_raise_string(val)
        reg = re.compile(r'^[A-Za-z0-9_]*$')
        if not reg.match(val):
            raise ValueError(f"String may oncly consist of digits, underscore_ or word chars, got {val}")
        
    def check_raise_string_only_abc123_extentions(val: str):
        """Checks the strings to only allow examples below
        words
        digits
        2024
        word_digits_and_underscores_1999
        a_video.mp4
        textfiles.txt
        other_extensions.sql
        """
        ValueHelper.check_raise_string(val)
        reg = re.compile(r'^[A-Za-z0-9_\-]+(\.[A-Za-z0-9_]+)?$')
        if not reg.match(val):
            raise ValueError(f"String may oncly consist of digits, underscore_ or word chars or file_extensions got {val}")