# from flask_sqlalchemy import SQLAlchemy
from repository.db import db
from sqlalchemy import CheckConstraint
from sqlalchemy.dialects.mysql import SMALLINT, JSON
from sqlalchemy.dialects.mysql import BINARY
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped
from datetime import datetime
import uuid

# TINYINT : -128 > 128
# SMALLINT : -32768 > 32767

def generate_uuid():
    return uuid.uuid4().bytes  # 16-byte binary

class DomainObject(db.Model):
    # Abstract does not create a table
    __abstract__ = True

    id = db.Column(BINARY(16), primary_key=True, default=generate_uuid)

    createdAt = db.Column(db.DateTime, nullable=False, default=datetime.now)
    updatedAt = db.Column(db.DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    def uuid_str(self):
        return str(uuid.UUID(bytes=self.id)) if self.id else None

    # ---- public API (do NOT override) ----
    # First DomainObject.to_dict() is called
    # Then Subclass.to_dict() is called
    def to_dict(self):
        data = self._to_dict() or {}

        data.update({
            # "id": self.uuid_str(),
            'id': self.id,
            'createdAt': self.createdAt.isoformat() if self.createdAt else None,
            'updatedAt': self.updatedAt.isoformat() if self.updatedAt else None,
        })
        return data

    # ---- subclass hook (must override) ----
    def _to_dict(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _to_dict()"
        )

class Account(DomainObject):
    __tablename__ = 'Accounts'
    email = db.Column(db.String(255), nullable=False, unique=True)
    firstName = db.Column(db.String(127), nullable=False)
    lastName = db.Column(db.String(127), nullable=False)
    passwordHash = db.Column(db.String(255), nullable=False)
    salt = db.Column(db.String(255), nullable=False)
    lastLogin = db.Column(db.DateTime, nullable=True)
    mfaEnabled = db.Column(db.Boolean, nullable=False, default=False)
    mfaCode = db.Column(db.String(6), nullable=True)
    mfaCodeExpires = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            'email': self.email,
            'firstName': self.firstName,
            'lastName': self.lastName,
            'lastLogin': self.lastLogin.isoformat() if self.lastLogin else None,
            'createdAt': self.createdAt.isoformat() if self.createdAt else None,
            'updatedAt': self.updatedAt.isoformat() if self.updatedAt else None,
            'mfaEnabled': self.mfaEnabled,
        }

class Folder(DomainObject):
    __tablename__ = 'Folders'
    name = db.Column(db.String(127), nullable=False)
    parentId = db.Column(BINARY(16), db.ForeignKey('Folders.id', ondelete='CASCADE'), nullable=True)
    parent = db.relationship('Folder', remote_side='Folder.id', backref='children', lazy='joined')
    videos = db.relationship('Video', backref='folder', lazy='dynamic') # Loaded lazily, so videoIDs are accecible, but full fetch only when explicitly asked

    # Define a composite unique constraint
    __table_args__ = (
        db.UniqueConstraint('name', 'parentId', name='_name_parent_unique_constraint'),
    )

    def to_dict(self):
        return {
            'name': self.name,
            'parentId' : self.parentId,
            'children': [child.id for child in self.children],
            'videoIds': [video.id for video in self.videos]
        }

class Source(DomainObject):
    __tablename__ = 'Sources'
    name = db.Column(db.String(127), nullable=False)

class CompetitionInfo(DomainObject):
    __tablename__ = 'CompetitionInfo'
    info = db.Column(db.String(255), nullable=False)
    year = db.Column(db.Integer, nullable=False)

class Tag(DomainObject):
    __tablename__ = 'Tags'
    name = db.Column(db.String(127), nullable=False)
    keywords = db.Column(db.String(511), nullable=True)

    tagGroupId = db.Column(BINARY(16), db.ForeignKey('TagGroups.id'), nullable=True)

class TagGroup(DomainObject):
    __tablename__ = 'TagGroups'
    name = db.Column(db.String(127), nullable=False, unique=True)
    parentId = db.Column(BINARY(16), db.ForeignKey('TagGroups.id', ondelete='CASCADE'), nullable=True)
    parent = db.relationship('TagGroup', remote_side='TagGroup.id', backref='children', lazy='joined')
    tags = db.relationship('Tag', backref='group', lazy=True)

# Association table for Video <-> Tag (Many-to-Many)
video_tag = db.Table('video_tag',
    db.Column('videoId', BINARY(16), db.ForeignKey('Videos.id', ondelete='CASCADE'), primary_key=True),
    db.Column('tagId', BINARY(16), db.ForeignKey('Tags.id', ondelete='CASCADE'), primary_key=True)
)

class Video(DomainObject):
    __tablename__ = 'Videos'
    folderId = db.Column(BINARY(16), db.ForeignKey('Folders.id', ondelete='CASCADE'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    frameLength = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    fps = db.Column(db.Float, nullable=False)
    training = db.Column(db.Boolean, nullable=False)
    qualitative = db.Column(db.Boolean, nullable=False)
    obstruction = db.Column(db.Boolean, nullable=False)
    private = db.Column(db.Boolean, nullable=False, default=False)
    source = db.Column(db.Integer, nullable=True)
    sourceInfo = db.Column(db.String(255), nullable=True)
    completed_skill_labels = db.Column(db.Boolean, nullable=False, default=False)
    competition = db.Column(BINARY(16), db.ForeignKey('CompetitionInfo.id', ondelete='CASCADE'))
    judgeDiffScore = db.Column(db.Float, nullable=True)

    frameLabels = db.relationship('FrameLabel', backref='video', lazy='joined')
    tags = db.relationship('Tag', secondary=video_tag, backref='videos', lazy='joined')

    __table_args__ = (
        db.UniqueConstraint('name', 'folderId', name='_name_folder_unique_constraint'),
    )

    def to_dict(self):
        return {
            'folderId' : self.folderId,
            'name' : self.name,
            'frameLength': self.frameLength,
            'width' : self.width,
            'height' : self.height,
            'fps' : self.fps,
            'training' : self.training,
            'qualitative' : self.qualitative,
            'obstruction' : self.obstruction
        }

# Exception to DomainObject (for easing out on training code)
class FrameLabelType(db.Model):
    __tablename__ = 'FrameLabelTypes'
    id = db.Column(db.Integer, primary_key=True)
    info = db.Column(db.String(127))

class FrameLabel(DomainObject):
    __tablename__ = 'FrameLabels'
    videoId = db.Column(BINARY(16), db.ForeignKey('Videos.id', ondelete='CASCADE'), nullable=False)
    frameNr = db.Column(SMALLINT(unsigned=True), nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    width = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    jumperVisible = db.Column(db.Boolean, nullable=False, default=True)
    labeltype = db.Column(db.Integer, db.ForeignKey('FrameLabelTypes.id', ondelete='CASCADE'), nullable=False, default=1)
    labeldate = db.Column(db.Date, default=lambda: datetime.now().date())
    labeltime = db.Column(db.Time, default=lambda: datetime.now().time())

    def to_dict(self):
        print(f"BAD BAD BAD FrameLabel to_dict called")
        return {
            'videoId' : self.videoId,
            'frameNr' : self.frameNr,
            'x' : self.x,
            'y' : self.y,
            'width' : self.width,
            'jumperVisible' : self.jumperVisible
        }

class TrainResultEpoch(DomainObject):
    __tablename__ = 'TrainResultsEpoch'
    trainResultId = db.Column(BINARY(16), db.ForeignKey('TrainResults.id', ondelete='CASCADE'), nullable=False)
    epoch = db.Column(SMALLINT(unsigned=True), nullable=False)
    validationResults = db.Column(MutableDict.as_mutable(JSON), nullable=False)

    def to_dict(self):
        return {
            'trainResultEpochId': self.id,
            'trainResultId': self.trainResultId,
            'epoch': self.epoch,
            'validationResults': self.validationResults,
        }

class TrainResult(DomainObject):
    __tablename__ = 'TrainResults'
    step = db.Column(db.String(50), nullable=False)
    recipeCode = db.Column(db.String(255), nullable=False)
    recipe = db.Column(MutableDict.as_mutable(JSON), nullable=False)

    bestEpoch = db.Column(SMALLINT(unsigned=True), nullable=False)
    revalidationResults = db.Column(MutableDict.as_mutable(JSON), nullable=False, default={})
    lastRevalidationTime = db.Column(db.DateTime, default=datetime.now)

    isBestOfAll = db.Column(db.Boolean, nullable=False)
    isBestOfRecipe = db.Column(db.Boolean, nullable=False)
    isBestOfArchitecture = db.Column(db.Boolean, nullable=False)

    isTestrun = db.Column(db.Boolean, nullable=False)
    trainEnd = db.Column(db.DateTime, nullable=True)

    epochs = db.relationship(
        TrainResultEpoch,
        backref='train_result',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )

    def to_dict(self):
        return {
            'step' : self.step,
            'recipeCode' : self.recipeCode,
            'recipe' : self.recipe,
            'bestEpoch' : self.bestEpoch,
            'revalidationResults' : self.revalidationResults,
            'lastRevalidationTime' : self.lastRevalidationTime,
            'isBestOfAll' : self.isBestOfAll,
            'isBestOfRecipe' : self.isBestOfRecipe,
            'isBestOfArchitecture' : self.isBestOfArchitecture,
            'isTestrun' : self.isTestrun,
            'trainEnd' : self.trainEnd,
            'epochs': {
                e.epoch: e.to_dict()
                for e in self.epochs.order_by(TrainResultEpoch.epoch).all()
            }
        }

class Skill(DomainObject):
    __tablename__ = 'Skills'
    videoId = db.Column(BINARY(16), db.ForeignKey('Videos.id', ondelete='CASCADE'), nullable=False)
    frameStart = db.Column(db.Integer, nullable=False)
    frameEnd = db.Column(db.Integer, nullable=False)
    skillinfo = db.Column(MutableDict.as_mutable(JSON), nullable=False)

# skillinfo_example = {
#     "composition1": [
#         {
#             "GeneralProperties": {"foo": "bar"},
#             "StartProperties": {},
#             "StageProperties": {
#                 "1": {"temp": 100, "pressure": 5},
#                 "2": {"humidity": 50}
#             }
#         }
#     ]
# }

class Jobs(DomainObject):
    __tablename__ = 'Jobs'
    type = db.Column(db.String(30), nullable=False)
    step = db.Column(db.String(127), nullable=False)
    job_arguments = db.Column(MutableDict.as_mutable(JSON), nullable=False)
    status = db.Column(db.String(30), nullable=False)
    status_details = db.Column(db.String(127))

class Layer(DomainObject):
    __tablename__ = 'Layers'
    name = db.Column(db.String(50), nullable=False)
    type = db.Column(db.String(15), nullable=False)
    min = db.Column(db.Float, nullable=True)
    max = db.Column(db.Float, nullable=True)
    step = db.Column(db.Float, nullable=True)

    categories = db.relationship('LayerValue', backref='layer', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name' : self.name,
            'type' : self.type,
            'min' : self.min,
            'max' : self.max,
            'step' : self.step,
            'categories': [c.to_dict() for c in sorted(self.categories, key=lambda c: c.name)]
        }

class LayerValue(DomainObject):
    __tablename__ = 'LayerValues'
    layerId = db.Column(BINARY(16), db.ForeignKey('Layers.id', ondelete='CASCADE'), nullable=False)
    name = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name
        }

class LayerComposition(DomainObject):
    __tablename__ = 'LayerComposition'
    compositionName = db.Column(db.String(50), nullable=False)
    stage = db.Column(db.Integer, CheckConstraint('stage >= -1'), nullable=True)
    layerId = db.Column(BINARY(16), db.ForeignKey('Layers.id', ondelete='CASCADE'), nullable=False)
    layer : Mapped[Layer] = db.relationship('Layer', backref='compositions', remote_side=[Layer.id],  lazy='joined')
    defaultValue = db.Column(db.String(15), nullable=True)
    focussed = db.Column(db.Boolean, nullable=False, default=True)

    def defaultValueConvert(self, value: str, layer: Layer):
        if value is None:
            return value
        match layer.type:
            case 'boolean':
                return True if value in ['true', 'True', 1, '1'] else False
            case 'categorical':
                return int(value)
            case 'numerical':
                return float(value)

    def to_dict(self):
        return {
            'id': self.id,
            'compositionName' : self.compositionName,
            'stage' : self.stage,
            'layer' : self.layer.to_dict(),
            'defaultValue': self.defaultValueConvert(self.defaultValue, self.layer),
            'focussed': self.focussed,
        }

