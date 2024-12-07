class ServiceHelper:
    def check_raise_id(id: int):
        if id <= 0:
            raise ValueError(f"Id must be strict positive integer, got {id}")
        
    def check_raise_string(val: str):
        if val.isspace():
            raise ValueError(f"Strings may not consist of spaces spaces")