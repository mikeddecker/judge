import re

class ValueHelper:
    def check_raise_id(id: int):
        if not isinstance(id, int):
            raise ValueError(f"Id must be of type {int}")
        if id <= 0:
            raise ValueError(f"Id must be strict positive integer, got {id}")
        
    def check_raise_string(val: str):
        if val is None:
            raise ValueError(f"Strings may not be none")
        if val.isspace() or val == "":
            raise ValueError(f"Strings may not be empty")
    
    def check_raise_string_only_abc123(val: str):
        ValueHelper.check_raise_string(val)
        reg = re.compile(r'^[A-Za-z0-9_]*$')
        if not reg.match(val):
            raise ValueError(f"String may oncly consist of digits, underscore_ or word chars, got {val}")