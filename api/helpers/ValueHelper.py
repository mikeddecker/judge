import re

MAX_FRAMENR = 65535

class ValueHelper:
    def check_raise_id(id: int):
        if not isinstance(id, int):
            raise ValueError(f"Id must be of type {int}, got {id}")
        if id <= 0:
            raise ValueError(f"Id must be strict positive integer, got {id}")
    
    def check_raise_frameNr(frameNr: int):
        if not isinstance(frameNr, int):
            raise ValueError(f"FrameNr must be of type {int}, got {frameNr}")
        if frameNr < 0:
            raise ValueError(f"FrameNr must be positive integer, got {frameNr}")
        if frameNr > MAX_FRAMENR:
            raise ValueError(f"FrameNr may max be {MAX_FRAMENR}, got {frameNr}")

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
        reg = re.compile(r'^[A-Za-z0-9_\-]*$')
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
        
    def check_float_between_0_and_1_inclusive(val: float):
        """
        Checks if the value is in the interval [0, 1]
        """
        if not isinstance(val, (int, float)):
            raise ValueError(f"val is not an int or float, got {type(val)}")
        if val < 0 or val > 1:
            raise ValueError(f"Value must in interval [0,1], got {val}")