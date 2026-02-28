from helpers.ValueHelper import ValueHelper
from uuid import UUID

class SkillType:
    PROPERTIES = [
        'Id', 'Name',
    ]

    def __init__(self, id: UUID, name: str = None):
        """Plain & simple class, id & name (frog, highfrog, pushup, jump...)"""
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

    def __setId(self, id: UUID):
        ValueHelper.check_raise_uuid(id)
        if hasattr(self, 'Id') and self.Id is not None:
            raise AttributeError(f"Cannot modify Id once it is set")
        assert id is not None, "Id can not be None"
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

