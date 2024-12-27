from typing import Dict, Set

from .folder import Folder
from .frameinfo import Frame
from .skill import Skill
from helpers.ValueHelper import ValueHelper

class VideoInfo:
    PROPERTIES = ["Frames", "Skills", "Id", "Name", "Folder"]
    # Frame does not 
    Frames = Dict[int, Frame] # Key = frameId, value is Frame
    Skills = Set[Skill]

    def __init__(self, id: int, name: str, folder: Folder):
        self.Frames: self.Frames = {}  # Initialize frames as an empty dictionary
        self.Skills: self.Skills = set()  # Initialize skills as an empty set

        self.__setId(id)
        self.__setName(name)
        self.__setFolder(folder)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name == 'Id':
                self.__setId(value)
            if name == 'Name':
                self.__setName(value)
            if name == 'Folder':
                self.__setFolder(value)
            # if name in ["Name", "Folder"]:
            #     raise AttributeError(f"Cannot modify '{name}' once it is set")
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

    def __setName(self, name : str):
        if hasattr(self, 'Name') and self.Name is not None:
            raise AttributeError(f"Cannot modify Name once it is set")
        if not name or name.isspace():
            raise ValueError("Name may not be an empty string")
        name = name.strip()
        if name.__contains__(" "):
            raise ValueError(f"Name may not contain spaces, got {name}")
        ValueHelper.check_raise_string_only_abc123_extentions(name)

        object.__setattr__(self, 'Name', name)
    
    def __setFolder(self, folder):
        if hasattr(self, 'Folder') and self.Folder is not None:
            raise AttributeError(f"Cannot modify Folder once it is set")
        if folder is None:
            raise ValueError(f"Folder is None")
        if not isinstance(folder, Folder):
            raise ValueError(f"folder is not a {Folder}, got instead {type(folder)}")
        # Set the Folder attribute, avoiding recursion by using object.__setattr__.
        object.__setattr__(self, 'Folder', folder)

    # Section : Frame functions

    # Section : Skill functions

    def add_skill(self, skill: Skill):
        if skill is None:
            raise ValueError("Skill may not be None")
        if skill in self.Skills:
            raise ValueError(f"Skill {skill} is already in the list.")
        self.Skills.add(skill)

    # TODO : update when videoinfo metadata is extended
    def __eq__(self, value : object):
        if not isinstance(value, VideoInfo):
            raise ValueError(f"Value not a {VideoInfo} got {type(value)} instead")
        
        # Typehint
        other : Folder = value
        
        # Check if both Ids are set
        if hasattr(self, "Id") and hasattr(other, "Id"):
            return (
                self.Name == other.Name and 
                self.Folder == other.Folder and 
                self.Id == other.Id
            )
        elif not hasattr(self, "Id") and not hasattr(other, "Id"):
            return (
                self.Name == other.Name and 
                self.Folder == other.Folder
            )
