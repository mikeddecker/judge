import os
from typing import Optional
from helpers.ValueHelper import ValueHelper

class Folder:
    PROPERTIES = ["Id", "Name", "Parent"]
    def __init__(self, id: int, name: str, parent: Optional['Folder'] = None):
        self.__setId(id)
        self.__setName(name)
        self.__setParent(parent)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name == 'Id':
                self.__setId(value)
            if name == 'Name':
                self.__setName(value)
            if name == "Parent":
                self.__setParent(value)
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def __setId(self, id: int):
        ValueHelper.check_raise_id(id)
        if hasattr(self, 'Id') and self.Id is not None:
            raise AttributeError(f"Cannot modify Id once it is set")
        if id is None or id <= 0:
            raise ValueError("Id must be strict positive")

        object.__setattr__(self, 'Id', id)


    def __setName(self, name : str):
        if hasattr(self, 'Name') and self.Name is not None:
            raise AttributeError(f"Cannot modify Name once it is set")
        if not name or name.isspace():
            raise ValueError("Name may not be an empty string")
        name = name.strip()
        if name.__contains__(" "):
            raise ValueError(f"Name may not contain spaces, got {name}")
        ValueHelper.check_raise_string_only_abc123(name)

        object.__setattr__(self, 'Name', name)
    
    def __setParent(self, parent):
        if hasattr(self, 'Parent') and self.Parent is not None:
            raise AttributeError(f"Cannot modify Parent once it is set")
        if parent is not None and not isinstance(parent, Folder):
            raise ValueError(f"parent is not a {Folder}, got instead {type(parent)}")

        # Set the Parent attribute, avoiding recursion by using object.__setattr__.
        object.__setattr__(self, 'Parent', parent)
        
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