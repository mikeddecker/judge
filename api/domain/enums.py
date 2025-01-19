from enum import Enum
class DDtype(Enum):
    DOUBLEDUTCH = 1
    SINGLEDUTCH = 2
    IRISH = 3
    CHINESEWHEEL = 4
    TRANSITION = 5

class SkillType(Enum):
    # Powers
    PUSHUP = 1
    FROG = 2
    HIGHFROG = 3
    SPLIT = 4

    # Gyms
    WEBSTER = 5
    SALTO = 6
    SUICIDE = 7


class Turner(Enum):
    CROSS = 1
    CROUGER = 2
    EB = 3
    TS = 4
    AS = 5
    JUMPOVER = 6
    CROSS_AW = 7
    JUMPOVER_TOAD = 8
    