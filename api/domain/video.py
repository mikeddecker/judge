# from typing import Optional
from folder import Folder

class Video:
    def __init__(self, id: int, name: str, folder: Folder):
        # Initialize the instance with values (this won't change the class-level variables)
        self.__setId(id)
        if not name or name.isspace():
            raise ValueError("Name may not be an empty string")
        if folder is not None and not isinstance(folder, Folder):
            raise ValueError(f"parent must be of type {Folder}, got {type(folder)}")
        self.Name = name
        self.Folder = folder

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name == 'Id':
                self.__setId(value)
            if name in ["Name", "Folder"]:
                raise AttributeError(f"Cannot modify '{name}' once it is set")
        super().__setattr__(name, value)

    def __setId(self, id):
        if hasattr(self, 'Id') and self.Id is not None:
            raise AttributeError(f"Cannot modify Id once it is set")
        if id is None or id <= 0:
            raise ValueError("Id must be strict positive")
        self.Id = id

