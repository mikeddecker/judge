from helpers.ValueHelper import ValueHelper

class Skill:
    PROPERTIES = [
        'Id', 'SkillInfo',
        'FrameStart', 'FrameEnd'
    ]
    Id: int
    FrameStart: int
    FrameEnd: int # Included --> [Start, End]

    def __init__(
            self,
            id: int,
            skillinfo: dict,
            start: int = None, 
            end: int = None
        ):
        """Either provide dotted name or parts (rotations, skilltype, onehanded, turners...)"""
        self.__setId(id)
        self.__setSkillinfo(skillinfo)
        self.__setFrameStart(start)
        self.__setFrameEnd(end)
            
    def __setattr__(self, name, value):
        if hasattr(self, name):
            if name == 'Id':
                self.__setId(value)
            # Prevent setting immutable attributes after it is set in __init__
            if name in ["Folder"]:
                raise AttributeError(f"Cannot modify '{name}' once it is set")
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

    def __setSkillinfo(self, info: dict):
        """Containing keys: column name, value: categoric, numeric, categoric numeric
        Also has key: Tablename, value: tablename
        Maybe rather:
        'Skill' : Tuple((Skillspecification_TablenameDiscipline_)Skill, str, None),
        '
        """
        assert info is not None, "SkillInfo may not be None"
        assert type(info) == dict, f"Info must be of type Dict, got {type(info)} - {info}"
        assert len(info.keys()) >= 1

        # TODO : new checks?
        self.SkillInfo = info

    def __setFrameStart(self, start: int):
        if not start or not isinstance(start, int) or start < 0:
            raise ValueError("Starting frame must be an integer bigger than 0")
        self.FrameStart = start
    
    def __setFrameEnd(self, end: int):
        if not end or not isinstance(end, int) or end < 0:
            raise ValueError("Starting frame must be an integer bigger than 0")
        self.FrameEnd = end
    
    def __str__(self):
        return str(self.to_dict())
    
    def to_dict(self):
        return {
            'Id' : self.Id,
            'Skillinfo' : self.SkillInfo,
            'FrameStart' : self.FrameStart,
            'FrameEnd' : self.FrameEnd,
        }

