class ServiceHelper:
    def check_raise_id(id: int):
        if id <= 0:
            raise ValueError(f"Id must be strict positive integer, got {id}")
        
    def check_raise_string(val: str):
        if val is None:
            raise ValueError(f"Strings may not be none")
        if val.isspace() or val == "":
            raise ValueError(f"Strings may not be empty")