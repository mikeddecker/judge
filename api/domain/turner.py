from helpers.ValueHelper import ValueHelper

class Turner:
    PROPERTIES = [
        'Id', 'Name',
    ]

    def __init__(self, id: int, name: str = None):
        """Plain & simple class, id & name (cross, crouger, EB, normal...)"""
        self.__setId(id)
        self.__setName(name)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            if name == 'Id':
                self.__setId(value)
            if name == 'Name':
                self.__setDottedName(value)
            # Prevent setting immutable attributes after it is set in __init__
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def __setId(self, id: int):
        ValueHelper.check_raise_id(id)
        if hasattr(self, 'Id') and self.Id is not None:
            raise AttributeError(f"Cannot modify Id once it is set")
        if id is None or id <= 0:
            raise ValueError("Id must be strict positive")
        self.Id = id

    def __setName(self, name:str):
        if hasattr(self, 'Name') and self.Name is not None:
            raise AttributeError(f"Cannot modify Name once it is set")
        if not name or name.isspace():
            raise ValueError("Name may not be an empty string")
        self.Name = name

    def __str__(self):
        return str(self.to_dict())
    
    def to_dict(self):
        return {
            'Id' : self.Id,
            'Name' : self.Name,
        }