# from flask_sqlalchemy import SQLAlchemy
from repository.db import db
from sqlalchemy import CheckConstraint
from sqlalchemy.dialects.mysql import SMALLINT, JSON
from sqlalchemy.ext.mutable import MutableDict
from datetime import datetime

class Folder(db.Model):
    __tablename__ = 'Folders'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(127), nullable=False)
    parentId = db.Column(db.Integer, db.ForeignKey('Folders.id', ondelete='CASCADE'), nullable=True)
    parent = db.relationship('Folder', remote_side=[id], backref='children', lazy='joined')
    videos = db.relationship('Video', backref='folder', lazy='dynamic') # Loaded lazily, so videoIDs are accecible, but full fetch only when explicitly asked

    # Define a composite unique constraint
    __table_args__ = (
        db.UniqueConstraint('name', 'parentId', name='_name_parent_unique_constraint'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'parentId' : self.parentId,
            'children': [child.id for child in self.children],
            'videoIds': [video.id for video in self.videos] 
        }
    
class Source(db.Model):
    __tablename__ = 'Sources'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(127), nullable=False)

class CompetitionInfo(db.Model):
    __tablename__ = 'CompetitionInfo'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    info = db.Column(db.String(255), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    
class TagGroup(db.Model):
    __tablename__ = 'TagGroups'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(127), nullable=False, unique=True)
    parentId = db.Column(db.Integer, db.ForeignKey('TagGroups.id', ondelete='CASCADE'), nullable=True)
    parent = db.relationship('TagGroup', remote_side=[id], backref='children', lazy='joined')
    tags = db.relationship('Tag', backref='group', lazy=True)

class Tag(db.Model):
    __tablename__ = 'Tags'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(127), nullable=False)
    keywords = db.Column(db.String(511), nullable=True)

    tagGroupId = db.Column(db.Integer, db.ForeignKey('TagGroups.id'), nullable=True)

# Association table for Video <-> Tag (Many-to-Many)
video_tag = db.Table('video_tag',
    db.Column('videoId', db.Integer, db.ForeignKey('Videos.id', ondelete='CASCADE'), primary_key=True),
    db.Column('tagId', db.Integer, db.ForeignKey('Tags.id', ondelete='CASCADE'), primary_key=True)
)

class Video(db.Model):
    __tablename__ = 'Videos'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    folderId = db.Column(db.Integer, db.ForeignKey('Folders.id', ondelete='CASCADE'), nullable=False)
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
    competition = db.Column(db.Integer, db.ForeignKey('CompetitionInfo.id', ondelete='CASCADE'))
    judgeDiffScore = db.Column(db.Float, nullable=True)

    frameLabels = db.relationship('FrameLabel', backref='video', lazy='joined')
    tags = db.relationship('Tag', secondary=video_tag, backref='videos', lazy='joined')

    __table_args__ = (
        db.UniqueConstraint('name', 'folderId', name='_name_folder_unique_constraint'),
    )

    def to_dict(self):
        return {
            'id': self.id,
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
    
class FrameLabelType(db.Model):
    __tablename__ = 'FrameLabelTypes'
    id = db.Column(db.Integer, primary_key=True)
    info = db.Column(db.String(127))
    
class FrameLabel(db.Model):
    __tablename__ = 'FrameLabels'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    videoId = db.Column(db.Integer, db.ForeignKey('Videos.id', ondelete='CASCADE'), nullable=False)
    frameNr = db.Column(SMALLINT(unsigned=True), nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    width = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    jumperVisible = db.Column(db.Boolean, nullable=False, default=True)
    labeltype = db.Column(db.Integer, db.ForeignKey('FrameLabelTypes.id', ondelete='CASCADE'), nullable=False, default=1)
    labeldatetime = db.Column(db.DateTime, default=datetime.now)
    labeldate = db.Column(db.Date, default=lambda: datetime.now().date())
    labeltime = db.Column(db.Time, default=lambda: datetime.now().time())

    def to_dict(self):
        return {
            'videoId' : self.videoId,
            'frameNr' : self.frameNr,
            'x' : self.x,
            'y' : self.y,
            'width' : self.width,
            'jumperVisible' : self.jumperVisible
        }
    
class TrainResult(db.Model):
    __tablename__ = 'TrainResults'
    modelname = db.Column(db.String(127), nullable=False)
    train_date = db.Column(db.Integer, nullable=False)
    epoch = db.Column(SMALLINT(unsigned=True), nullable=False)
    iou = db.Column(db.Float, nullable=False)
    loss = db.Column(db.Float, nullable=False)
    val_iou = db.Column(db.Float, nullable=False)
    val_loss = db.Column(db.Float, nullable=False)

    __table_args__ = (
        db.PrimaryKeyConstraint('modelname', 'epoch', name='_modelname_epoch_pk_constraint'),
    )

class TrainResultSkill(db.Model):
    __tablename__ = 'TrainResultsSkills'
    modelname = db.Column(db.String(127), nullable=False)
    train_date = db.Column(db.Integer, nullable=False)
    epoch = db.Column(SMALLINT(unsigned=True), nullable=False)
    loss = db.Column(db.Float, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    val_loss = db.Column(db.Float, nullable=False)
    val_accuracy = db.Column(db.Float, nullable=False)
    losses_and_metrics = db.Column(MutableDict.as_mutable(JSON), nullable=False)

    __table_args__ = (
        db.PrimaryKeyConstraint('modelname', 'epoch', name='_modelname_epoch_pk_constraint'),
    )

class Skill(db.Model):
    __tablename__ = 'Skills'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    videoId = db.Column(db.Integer, db.ForeignKey('Videos.id', ondelete='CASCADE'), nullable=False)
    frameStart = db.Column(db.Integer, nullable=False)
    frameEnd = db.Column(db.Integer, nullable=False)
    skillinfo = db.Column(MutableDict.as_mutable(JSON), nullable=False)
    labeldate = db.Column(db.DateTime, default=datetime.now)
    updated = db.Column(db.DateTime, default=datetime.now)

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

class Jobs(db.Model):
    __tablename__ = 'Jobs'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type = db.Column(db.String(30), nullable=False)
    step = db.Column(db.String(127), nullable=False)
    job_arguments = db.Column(MutableDict.as_mutable(JSON), nullable=False)
    request_time = db.Column(db.DateTime, default=datetime.now)
    status = db.Column(db.String(30), nullable=False)
    status_details = db.Column(db.String(127))

class LayerProperty(db.Model):
    __tablename__ = 'LayerProperties'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50), nullable=False)
    type = db.Column(db.String(15), nullable=False)
    min = db.Column(db.Float, nullable=True)
    max = db.Column(db.Float, nullable=True)
    step = db.Column(db.Float, nullable=True)
    creationDate = db.Column(db.DateTime, default=datetime.now)
    lastUpdated = db.Column(db.DateTime, default=datetime.now)

    # categories = db.relationship('LayerPropertyValue', backref='property', lazy='dynamic')
    categories = db.relationship('LayerPropertyValue', backref='property', lazy=True)

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

class LayerPropertyValue(db.Model):
    __tablename__ = 'LayerPropertyValues'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    propertyId = db.Column(db.Integer, db.ForeignKey('LayerProperties.id', ondelete='CASCADE'), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    creationDate = db.Column(db.DateTime, default=datetime.now)
    lastUpdated = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name
        }

class LayerComposition(db.Model):
    __tablename__ = 'LayerComposition'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    compositionName = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(50), nullable=True)
    stage = db.Column(db.Integer, CheckConstraint('stage >= -1'), nullable=True)
    propertyId = db.Column(db.Integer, db.ForeignKey('LayerProperties.id', ondelete='CASCADE'), nullable=False)
    property = db.relationship('LayerProperty', backref='compositions', lazy='joined')
    defaultValue = db.Column(db.String(15), nullable=True)
    focussed = db.Column(db.Boolean, nullable=False, default=True)
    creationDate = db.Column(db.DateTime, default=datetime.now)
    lastUpdated = db.Column(db.DateTime, default=datetime.now)

    def defaultValueConvert(self, value: str, property: LayerProperty):
        if value is None:
            return value
        match property.type:
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
            'name': self.name,
            'stage' : self.stage,
            'property' : self.property.to_dict(),
            'defaultValue': self.defaultValueConvert(self.defaultValue, self.property),
            'focussed': self.focussed,
        }

