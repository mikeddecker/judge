from domain.enums import DDtype
from domain.skilltype import SkillType
from domain.turner import Turner
from helpers.ValueHelper import ValueHelper

class Skill:
    PROPERTIES = [
        'Id', 'DottedName',
        'DisciplineInfo', 'skillinfo',
        'FrameStart', 'FrameEnd'
    ]
    FrameStart: int
    FrameEnd: int # Included --> [Start, End]

    Type: DDtype
    Rotations: int
    JumperSkill: SkillType
    OneHanded: bool
    Turntable: int # TT1 = quarter, TT3 = 3/4th, TT4 = full turn table
    BodyRotations: int # 0, 1 or 2: like crab to crab = 1
    Turner1: Turner
    Turner2: Turner

    def __init__(self,
                 disciplineConfig: dict, # Containing keys: column name, value: which foreign key to use (TODO: restrictions)
                 skillinfo: dict, # Containing keys: columns, values: skillIds (ints) or numeric values
                 start: int = None, end: int = None):
        """Either provide dotted name or parts (rotations, skilltype, onehanded, turners...)"""
        self.__setDisciplineConfig(disciplineConfig)
        self.__setSkillinfo(skillinfo)
        self.__setFrameStart(start)
        self.__setFrameEnd(end)
        self.__setDottedNameFromAttributes()
            

    def __setattr__(self, name, value):
        if hasattr(self, name):
            if name == 'Id':
                self.__setDisciplineId(value)
            if name == 'DottedName':
                self.__setDottedName(value)
            # Prevent setting immutable attributes after it is set in __init__
            if name in ["DottedName", "Folder"]:
                raise AttributeError(f"Cannot modify '{name}' once it is set")
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def __setDisciplineId(self, id: int):
        ValueHelper.check_raise_id(id)
        if hasattr(self, 'Id') and self.Id is not None:
            raise AttributeError(f"Cannot modify Id once it is set")
        if id is None or id <= 0:
            raise ValueError("Id must be strict positive")
        self.Id = id

    def __setDisciplineConfig(self, disciplineinfo: dict):
        """Containing keys: column name, value: categoric, numeric, categoric numeric
        Also has key: Tablename, value: tablename
        Maybe rather:
        'Skill' : Tuple((Skillspecification_TablenameDiscipline_)Skill, str, None),
        '
        """
        assert disciplineinfo is not None, "Disciplineinfo may not be None"
        assert type(disciplineinfo) == dict, "Disciplineinfo must be of type Dict"
        assert "Tablename" in disciplineinfo.keys(), "Tablename must be specified"
        assert type(disciplineinfo["Tablename"]) == str, "Tablename must be a string"
        ValueHelper.check_raise_string_only_abc123(disciplineinfo["Tablename"])
        assert len(disciplineinfo.keys()) > 1

        # TODO : other checks?
        self.DisciplineInfo = disciplineinfo

    def __setSkillinfo(self, info: dict):
        """Containing keys: column name, value: categoric, numeric, categoric numeric
        Also has key: Tablename, value: tablename
        Maybe rather:
        'Skill' : Tuple((Skillspecification_TablenameDiscipline_)Skill, str, None),
        '
        """
        assert info is not None, "SkillInfo may not be None"
        assert type(info) == dict, "Info must be of type Dict"
        assert len(info.keys()) >= 1
        for key, value in info.items():
            assert key in self.DisciplineInfo.keys(), f"Key {key} not listed in configuration of skill"
            assert isinstance(value, int), f"Value must be foreign key / unique key (integer) of the table"

        # TODO : other checks?
        self.SkillInfo = info


    def __setDottedName(self, dottedName:str):
        if hasattr(self, 'DottedName') and self.DottedName is not None:
            raise AttributeError(f"Cannot modify DottedName once it is set")
        if not dottedName or dottedName.isspace():
            raise ValueError("DottedName may not be an empty string")
        self.DottedName = dottedName

    def __setDDtype(self, ddtype: DDtype):
        if hasattr(self, 'Type') and self.Type is not None:
            raise AttributeError(f"Cannot modify DottedName once it is set")
        assert type(ddtype) == DDtype
        self.DottedName = ddtype

    def __setRotations(self, rotations: int):
        if hasattr(self, 'Rotations') and self.Rotations is not None:
            raise AttributeError(f"Cannot modify multiple once it is set")
        if not rotations:
            raise ValueError("Jumperskill can not be None")
        self.Rotations = rotations

    def __setJumperSkill(self, jumperskill: SkillType):
        if hasattr(self, 'JumperSkill') and self.DottedName is not None:
            raise AttributeError(f"Cannot modify jumperskill once it is set")
        if not jumperskill:
            raise ValueError("Jumperskill can not be None")
        self.JumperSkill = jumperskill
    
    def __setFrameStart(self, start: int):
        if not start or not isinstance(start, int) or start < 0:
            raise ValueError("Starting frame must be an integer bigger than 0")
        self.FrameStart = start
    
    def __setFrameEnd(self, end: int):
        if not end or not isinstance(end, int) or end < 0:
            raise ValueError("Starting frame must be an integer bigger than 0")
        self.FrameEnd = end
    
    def __setAttributesFromDottedName(self):
        raise NotImplementedError("FOEI MIKE FOEI")

    def __setDottedNameFromAttributes(self):
        self.__setDottedName(".".join([self.Type, self.Rotations, self.JumperSkill, self.OneHanded, self.Turner1, self.Turner2]))

    def __str__(self):
        return str(self.to_dict())
    
    def to_dict(self):
        return {
            'Id' : self.Id,
            'DottedName' : self.DottedName,
            'FrameStart' : self.FrameStart,
            'FrameEnd' : self.FrameEnd,
        }