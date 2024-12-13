import os
from typing import Optional

class Folder:
    PROPERTIES = ["Id", "Name", "Parent"]
    def __init__(self, id: int, name: str, parent: Optional['Folder'] = None):
        self.__setId(id)
        if not name or name.isspace():
            raise ValueError("Name may not be an empty string")
        if parent is not None and not isinstance(parent, Folder):
            raise ValueError(f"parent must be of type {Folder}, got {type(parent)}")
        self.Name = name
        self.Parent = parent

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name == 'Id':
                self.__setId(value)
            if name == "Parent" and self.Parent is None:
                super().__setattr__(name, value)
                return
            if name in ["Name", "Parent"]:
                raise AttributeError(f"Cannot modify '{name}' once it is set")
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def __setId(self, id):
        if hasattr(self, 'Id') and self.Id is not None:
            raise AttributeError(f"Cannot modify Id once it is set")
        if id is None or id <= 0:
            raise ValueError("Id must be strict positive")
        self.Id = id
    
    def get_relative_path(self):
        if self.Parent:
            return os.path.join(self.Parent.get_relative_path(), self.Name)
        return self.Name
    
    def __eq__(self, value : object):
        if not isinstance(value, Folder):
            raise ValueError(f"Value not a {Folder} got {type(value)} instead")
        
        # Typehint
        other : Folder = value
        
        # Check if both Ids are set
        if hasattr(self, "Id") and hasattr(other, "Id"):
            return (
                self.Name == other.Name and 
                self.Parent == other.Parent and 
                self.Id == other.Id
            )
        elif not hasattr(self, "Id") and not hasattr(other, "Id"):
            return (
                self.Name == other.Name and 
                self.Parent == other.Parent
            )