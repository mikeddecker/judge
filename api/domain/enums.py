from enum import Enum


class Discipline(Enum):
    DOUBLEDUTCH = 1


class DDtype(Enum):
    DOUBLEDUTCH = 1
    SINGLEDUTCH = 2
    IRISH = 3
    CHINESEWHEEL = 4
    TRANSITION = 5


class AccountType(str, Enum):
    ADMIN        = 'admin'
    USER         = 'user'
    GROUP        = 'group'
    TEAM         = 'team'
    ORGANISATION = 'organisation'


class GrantedTo(str, Enum):
    EVERYONE = 'everyone'
    ACCOUNT  = 'account'
    GROUP    = 'group'


class RelationshipType(str, Enum):
    FRIEND         = 'friend'
    MEMBER         = 'member'
    REPRESENTATIVE = 'representative'
    FOLLOWER       = 'follower'
    INDIVIDUAL     = 'individual'
