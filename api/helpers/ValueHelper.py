import re
# from repository.videoRepo import VideoRepository

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
        
    def check_raise_yt_url(val: str):
        """Checks if a given url is formatted as a yt url"""
        reg = re.compile(r'^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(?:-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|live\/|v\/)?)([\w\-]+)(\S+)?$')
        if not reg.match(val):
            raise ValueError(f"Not a valid yt url, got {val}")
        
    def check_raise_skillinfo_values(config: dict, skillinfo: dict, repo):
        """Checks whether the giving skillinfo corresponds with the giving config info"""
        assert isinstance(config, dict), f"Config is not a dict, got {config}"
        assert isinstance(skillinfo, dict), f"Skillinfo is not a dict, got {skillinfo}"
        assert len(config) > 0, f"Config can not be empty, got {config}"
        assert len(skillinfo) > 0, f"Skillinfo can not be empty, got {skillinfo}"

        # Check skillinfo values
        for key, value in config.items():
            if key != 'Tablename':
                assert key in skillinfo.keys(), f"Skillinfo does not provide info for {key}"
            if value[0] == "Numerical":
                min = value[1]
                max = value[2]
                assert isinstance(skillinfo[key], int), f"Skillspecification of {key} must be in integer, got {skillinfo[key]}"
                assert skillinfo[key] >= min and skillinfo[key] <= max, f"Skillinfo {key} must be between {min} and {max}, got {skillinfo[key]}"
            elif value[0] == "Categorical":
                assert isinstance(skillinfo[key], int), f"Skillspecification of {key} must be in integer, got {skillinfo[key]}"
                repo.exists_skillinfo(discipline=config["Tablename"], table_name_part=config[key][1], uc=skillinfo[key])
            elif value[0] == "Boolean":
                assert isinstance(skillinfo[key], bool), f"Boolean value {key} must be a boolean, got {skillinfo[key]}"

