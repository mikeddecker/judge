# from flask_sqlalchemy import SQLAlchemy
from repository.db import db
from sqlalchemy import CheckConstraint
from sqlalchemy.dialects.mysql import SMALLINT, JSON
from sqlalchemy.dialects.mysql import BINARY
from sqlalchemy.types import TypeDecorator
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped
from datetime import datetime
import uuid

GRANTED_TO_VALUES = ('everyone', 'account', 'group')
RELATIONSHIP_TYPE_VALUES = ('friend', 'member', 'representative', 'follower', 'individual')


def _uuid_to_str(value) -> str:
    """Safely convert a binary UUID (bytes or uuid.UUID) to a hyphenated string."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes):
        return str(uuid.UUID(bytes=value))
    return str(value)

# TINYINT : -128 > 128
# SMALLINT : -32768 > 32767

def generate_uuid():
    return uuid.uuid4().bytes  # 16-byte binary

class UUIDType(TypeDecorator):
    impl = BINARY(16)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if isinstance(value, uuid.UUID):
            return value.bytes
        if isinstance(value, str):
            return uuid.UUID(value).bytes
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return uuid.UUID(bytes=value)
        return value

class DomainObject(db.Model):
    # Abstract does not create a table
    __abstract__ = True

    id = db.Column(UUIDType, primary_key=True, default=generate_uuid)

    createdAt = db.Column(db.DateTime, nullable=False, default=datetime.now)
    updatedAt = db.Column(db.DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    def uuid_str(self):
        # Return canonical UUID string (with hyphens) for readability and compatibility
        return str(uuid.UUID(bytes=self.id)) if self.id else None

    # ---- public API (do NOT override) ----
    # First DomainObject.to_dict() is called
    # Then Subclass.to_dict() is called
    def to_dict(self):
        data = self._to_dict() or {}

        data.update({
            'id': self.uuid_str(),
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
    # Account type: user (default), group, team, organisation, admin
    accountType = db.Column(db.String(20), nullable=False, default='user')
    # For group/team/org accounts: the account that created/owns this account
    owner_id = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=True)

    def to_dict(self):
        return {
            'id': self.uuid_str(),
            'email': self.email,
            'firstName': self.firstName,
            'lastName': self.lastName,
            'lastLogin': self.lastLogin.isoformat() if self.lastLogin else None,
            'createdAt': self.createdAt.isoformat() if self.createdAt else None,
            'updatedAt': self.updatedAt.isoformat() if self.updatedAt else None,
            'mfaEnabled': self.mfaEnabled,
            'accountType': self.accountType,
        }

class Folder(DomainObject):
    __tablename__ = 'Folders'
    name = db.Column(db.String(127), nullable=False)
    parentId = db.Column(UUIDType, db.ForeignKey('Folders.id', ondelete='CASCADE'), nullable=True)
    parent = db.relationship('Folder', remote_side='Folder.id', backref='children', lazy='joined')
    videos = db.relationship('Video', backref='folder', lazy='dynamic') # Loaded lazily, so videoIDs are accecible, but full fetch only when explicitly asked

    is_train = db.Column(db.Boolean, nullable=False, default=True)

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

    tagGroupId = db.Column(UUIDType, db.ForeignKey('TagGroups.id'), nullable=True)

class TagGroup(DomainObject):
    __tablename__ = 'TagGroups'
    name = db.Column(db.String(127), nullable=False, unique=True)
    parentId = db.Column(UUIDType, db.ForeignKey('TagGroups.id', ondelete='CASCADE'), nullable=True)
    parent = db.relationship('TagGroup', remote_side='TagGroup.id', backref='children', lazy='joined')
    tags = db.relationship('Tag', backref='group', lazy=True)

# Association table for Video <-> Tag (Many-to-Many)
video_tag = db.Table('video_tag',
    db.Column('videoId', UUIDType, db.ForeignKey('Videos.id', ondelete='CASCADE'), primary_key=True),
    db.Column('tagId', UUIDType, db.ForeignKey('Tags.id', ondelete='CASCADE'), primary_key=True)
)

class Video(DomainObject):
    __tablename__ = 'Videos'
    folderId = db.Column(UUIDType, db.ForeignKey('Folders.id', ondelete='CASCADE'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    frameLength = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    fps = db.Column(db.Float, nullable=False)
    qualitative = db.Column(db.Boolean, nullable=False)
    obstruction = db.Column(db.Boolean, nullable=False)
    source = db.Column(db.Integer, nullable=True)
    sourceInfo = db.Column(db.String(255), nullable=True)
    completed_skill_labels = db.Column(db.Boolean, nullable=False, default=False)
    competition = db.Column(UUIDType, db.ForeignKey('CompetitionInfo.id', ondelete='CASCADE'))
    judgeDiffScore = db.Column(db.Float, nullable=True)
    private = db.Column(db.Boolean, nullable=False, default=True)
    training = db.Column(db.Boolean, nullable=False)
    is_train = db.Column(db.Boolean, nullable=False, default=True)

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
    videoId = db.Column(UUIDType, db.ForeignKey('Videos.id', ondelete='CASCADE'), nullable=False)
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
    trainResultId = db.Column(UUIDType, db.ForeignKey('TrainResults.id', ondelete='CASCADE'), nullable=False)
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
    videoId = db.Column(UUIDType, db.ForeignKey('Videos.id', ondelete='CASCADE'), nullable=False)
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

class ConflictLog(DomainObject):
    """Tracks first-write-wins conflicts in dual-primary replication
    
    Conflicts can be:
    1. Auto-resolved: Automatically kept winning version if non-critical field (logged but not notified)
    2. User-resolved: Critical fields (frame labels, skill data) require user inspection and approval
    """
    __tablename__ = 'ConflictLogs'
    
    entity_type = db.Column(db.String(50), nullable=False)  # e.g., 'Video', 'FrameLabel', 'Skill'
    entity_id = db.Column(UUIDType, nullable=False, index=True)  # ID of the conflicting entity
    
    # Winning update (kept)
    winning_account_id = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    winning_region = db.Column(db.String(50), nullable=False)
    winning_timestamp = db.Column(db.DateTime, nullable=False)
    winning_data = db.Column(MutableDict.as_mutable(JSON), nullable=False)  # Full winning entity state
    
    # Losing update (archived for audit)
    losing_account_id = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=True)
    losing_region = db.Column(db.String(50), nullable=False)
    losing_timestamp = db.Column(db.DateTime, nullable=False)
    losing_data = db.Column(MutableDict.as_mutable(JSON), nullable=False)  # Full losing entity state
    
    # Resolution tracking
    conflict_description = db.Column(db.String(255), nullable=False)  # Human-readable description
    auto_resolved = db.Column(db.Boolean, nullable=False, default=False)  # True if auto-resolved (no user action needed)
    is_resolved = db.Column(db.Boolean, nullable=False, default=False)  # True if user manually resolved
    resolved_by = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    resolution_note = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            'entity_type': self.entity_type,
            'entity_id': str(self.entity_id) if self.entity_id else None,
            'winning_account_id': str(self.winning_account_id) if self.winning_account_id else None,
            'winning_region': self.winning_region,
            'winning_data': self.winning_data,
            'losing_account_id': str(self.losing_account_id) if self.losing_account_id else None,
            'losing_region': self.losing_region,
            'losing_data': self.losing_data,
            'conflict_description': self.conflict_description,
            'auto_resolved': self.auto_resolved,
            'is_resolved': self.is_resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
        }

class DeletedVideo(DomainObject):
    """Tracks soft-deleted videos for 30-day recovery window before hard delete"""
    __tablename__ = 'DeletedVideos'
    
    video_id = db.Column(UUIDType, nullable=False, index=True, unique=True)  # Reference to original Video.id
    video_name = db.Column(db.String(255), nullable=False)  # Store video name for recovery context
    folder_id = db.Column(UUIDType, nullable=False)  # Store folder reference
    deleted_by = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    deleted_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    
    recovery_deadline = db.Column(db.DateTime, nullable=False)  # calculated as deleted_at + 30 days
    hard_delete_at = db.Column(db.DateTime, nullable=True)  # when hard delete executed
    
    region = db.Column(db.String(50), nullable=False)  # Region where deletion originated
    original_video_data = db.Column(MutableDict.as_mutable(JSON), nullable=False)  # Archived video metadata for restore
    storage_path = db.Column(db.String(511), nullable=False)  # Path to video files for restoration
    
    is_hard_deleted = db.Column(db.Boolean, nullable=False, default=False)
    hard_deleted_by = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=True)

    def to_dict(self):
        return {
            'video_id': str(self.video_id) if self.video_id else None,
            'video_name': self.video_name,
            'deleted_at': self.deleted_at.isoformat() if self.deleted_at else None,
            'recovery_deadline': self.recovery_deadline.isoformat() if self.recovery_deadline else None,
            'is_hard_deleted': self.is_hard_deleted,
        }

class Jobs(DomainObject):
    __tablename__ = 'Jobs'
    type = db.Column(db.String(30), nullable=False)
    step = db.Column(db.String(127), nullable=False)
    job_arguments = db.Column(MutableDict.as_mutable(JSON), nullable=False)
    status = db.Column(db.String(30), nullable=False)
    status_details = db.Column(db.String(127))
    job_category = db.Column(db.String(20), nullable=False, default='AI')  # 'AI', 'SYNC', or 'BACKUP'

class ExportJob(DomainObject):
    """Tracks data export requests for GDPR Article 20 compliance (right to data portability)
    
    Path A: On-demand export of account data as readable ZIP file
    Returns: 
      - Instant ZIP download (< 5 GB)
      - Async job with download link (>= 5 GB)
    """
    __tablename__ = 'ExportJobs'
    
    account_id = db.Column(UUIDType, db.ForeignKey('Accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Export request parameters
    include_metadata = db.Column(db.Boolean, nullable=False, default=True)  # Include metadata JSONs
    include_training_results = db.Column(db.Boolean, nullable=False, default=True)  # Include weights/models
    include_frames = db.Column(db.Boolean, nullable=False, default=False)  # Include frame-by-frame extracts (expensive)
    
    # Execution tracking
    status = db.Column(db.String(30), nullable=False, default='Pending')  # Pending, Processing, Completed, Failed
    estimated_size_gb = db.Column(db.Float, nullable=True)  # Estimated size at creation time
    actual_size_gb = db.Column(db.Float, nullable=True)  # Actual size after completion
    
    # Result
    download_url = db.Column(db.String(511), nullable=True)  # S3/local path to download ZIP
    file_path = db.Column(db.String(511), nullable=True)  # Filesystem path to generated ZIP
    
    # Expiration
    expires_at = db.Column(db.DateTime, nullable=True)  # When download link expires (default 7 days)
    downloaded_at = db.Column(db.DateTime, nullable=True)  # When file was actually downloaded
    
    # Error tracking
    error_message = db.Column(db.String(511), nullable=True)  # If status='Failed'
    
    # Audit
    requested_by = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)  # Account who requested (for audit)
    
    def to_dict(self):
        return {
            'id': self.uuid_str(),
            'account_id': str(self.account_id) if self.account_id else None,
            'status': self.status,
            'estimated_size_gb': self.estimated_size_gb,
            'actual_size_gb': self.actual_size_gb,
            'download_url': self.download_url,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'error_message': self.error_message,
            'createdAt': self.createdAt.isoformat() if self.createdAt else None,
        }

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
    layerId = db.Column(UUIDType, db.ForeignKey('Layers.id', ondelete='CASCADE'), nullable=False)
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
    layerId = db.Column(UUIDType, db.ForeignKey('Layers.id', ondelete='CASCADE'), nullable=False)
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


class AccountCapability(DomainObject):
    """Account capabilities — one row per account, managed by admin/subscription."""
    __tablename__ = 'AccountCapabilities'

    account_id = db.Column(UUIDType, db.ForeignKey('Accounts.id', ondelete='CASCADE'), nullable=False, unique=True)

    can_upload_video = db.Column(db.Boolean, nullable=False, default=False)
    can_edit_video = db.Column(db.Boolean, nullable=False, default=False)
    can_label_video = db.Column(db.Boolean, nullable=False, default=False)
    can_see_video_actions = db.Column(db.Boolean, nullable=False, default=False)
    can_see_tags = db.Column(db.Boolean, nullable=False, default=False)
    can_edit_tags = db.Column(db.Boolean, nullable=False, default=False)
    can_see_video_tags = db.Column(db.Boolean, nullable=False, default=False)
    can_see_video_labels = db.Column(db.Boolean, nullable=False, default=False)
    can_train_model = db.Column(db.Boolean, nullable=False, default=False)
    can_export_model = db.Column(db.Boolean, nullable=False, default=False)
    can_manage_members = db.Column(db.Boolean, nullable=False, default=False)
    can_manage_representatives = db.Column(db.Boolean, nullable=False, default=False)
    can_invite_users = db.Column(db.Boolean, nullable=False, default=False)

    max_video_uploads = db.Column(db.Integer, nullable=False, default=100)
    max_video_size_mb = db.Column(db.Integer, nullable=False, default=2048)
    max_video_duration_seconds = db.Column(db.Integer, nullable=False, default=120)
    max_storage_gb = db.Column(db.Integer, nullable=False, default=20)

    granted_by = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    granted_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    granted_until = db.Column(db.DateTime, nullable=True)
    granted_reason = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            'id': self.uuid_str(),
            'account_id': _uuid_to_str(self.account_id),
            'can_upload_video': self.can_upload_video,
            'can_edit_video': self.can_edit_video,
            'can_label_video': self.can_label_video,
            'can_see_video_actions': self.can_see_video_actions,
            'can_see_tags': self.can_see_tags,
            'can_edit_tags': self.can_edit_tags,
            'can_see_video_tags': self.can_see_video_tags,
            'can_see_video_labels': self.can_see_video_labels,
            'can_train_model': self.can_train_model,
            'can_export_model': self.can_export_model,
            'can_manage_members': self.can_manage_members,
            'can_manage_representatives': self.can_manage_representatives,
            'can_invite_users': self.can_invite_users,
            'max_video_uploads': self.max_video_uploads,
            'max_video_size_mb': self.max_video_size_mb,
            'max_video_duration_seconds': self.max_video_duration_seconds,
            'max_storage_gb': self.max_storage_gb,
            'granted_by': _uuid_to_str(self.granted_by),
            'granted_at': self.granted_at.isoformat() if self.granted_at else None,
            'granted_until': self.granted_until.isoformat() if self.granted_until else None,
            'granted_reason': self.granted_reason,
        }


class GroupMembership(DomainObject):
    """Many-to-many: accounts ↔ group accounts (accountType='group')."""
    __tablename__ = 'GroupMemberships'

    group_id = db.Column(UUIDType, db.ForeignKey('Accounts.id', ondelete='CASCADE'), nullable=False)
    account_id = db.Column(UUIDType, db.ForeignKey('Accounts.id', ondelete='CASCADE'), nullable=False)
    added_by = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    added_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        db.UniqueConstraint('group_id', 'account_id', name='_group_account_unique'),
    )

    def to_dict(self):
        return {
            'id': self.uuid_str(),
            'group_id': _uuid_to_str(self.group_id),
            'account_id': _uuid_to_str(self.account_id),
            'added_by': _uuid_to_str(self.added_by),
            'added_at': self.added_at.isoformat() if self.added_at else None,
        }


class AccessGrant(DomainObject):
    """Controls who can see or interact with the owner's content."""
    __tablename__ = 'AccessGrants'

    owner_id = db.Column(UUIDType, db.ForeignKey('Accounts.id', ondelete='CASCADE'), nullable=False)

    granted_to = db.Column(db.Enum(*GRANTED_TO_VALUES, name='granted_to_enum'), nullable=False)
    target_account_id = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=True)
    # target_group_id references a group account (accountType='group')
    target_group_id = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=True)

    # NULL = applies to all owner content
    video_id = db.Column(UUIDType, db.ForeignKey('Videos.id', ondelete='CASCADE'), nullable=True)
    folder_id = db.Column(UUIDType, db.ForeignKey('Folders.id', ondelete='CASCADE'), nullable=True)

    can_view = db.Column(db.Boolean, nullable=False, default=False)
    can_comment = db.Column(db.Boolean, nullable=False, default=False)
    can_label = db.Column(db.Boolean, nullable=False, default=False)
    can_download = db.Column(db.Boolean, nullable=False, default=False)

    relationship_type = db.Column(
        db.Enum(*RELATIONSHIP_TYPE_VALUES, name='relationship_type_enum'), nullable=True
    )

    granted_by = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    granted_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    granted_until = db.Column(db.DateTime, nullable=True)
    granted_reason = db.Column(db.Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "(granted_to = 'everyone' AND target_account_id IS NULL AND target_group_id IS NULL)"
            " OR (granted_to = 'account' AND target_account_id IS NOT NULL AND target_group_id IS NULL)"
            " OR (granted_to = 'group' AND target_group_id IS NOT NULL AND target_account_id IS NULL)",
            name='ck_access_grant_target',
        ),
        CheckConstraint(
            "NOT (video_id IS NOT NULL AND folder_id IS NOT NULL)",
            name='ck_access_grant_scope',
        ),
    )

    def to_dict(self):
        return {
            'id': self.uuid_str(),
            'owner_id': _uuid_to_str(self.owner_id),
            'granted_to': self.granted_to,
            'target_account_id': _uuid_to_str(self.target_account_id),
            'target_group_id': _uuid_to_str(self.target_group_id),
            'video_id': _uuid_to_str(self.video_id),
            'folder_id': _uuid_to_str(self.folder_id),
            'can_view': self.can_view,
            'can_comment': self.can_comment,
            'can_label': self.can_label,
            'can_download': self.can_download,
            'relationship_type': self.relationship_type,
            'granted_by': _uuid_to_str(self.granted_by),
            'granted_at': self.granted_at.isoformat() if self.granted_at else None,
            'granted_until': self.granted_until.isoformat() if self.granted_until else None,
            'granted_reason': self.granted_reason,
        }


class AccountBlock(DomainObject):
    """Block list — blocks always override access grants."""
    __tablename__ = 'AccountBlocks'

    blocker_id = db.Column(UUIDType, db.ForeignKey('Accounts.id', ondelete='CASCADE'), nullable=False)
    blocked_id = db.Column(UUIDType, db.ForeignKey('Accounts.id', ondelete='CASCADE'), nullable=False)
    blocked_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    reason = db.Column(db.Text, nullable=True)

    __table_args__ = (
        db.UniqueConstraint('blocker_id', 'blocked_id', name='_blocker_blocked_unique'),
        CheckConstraint('blocker_id != blocked_id', name='ck_block_no_self'),
    )

    def to_dict(self):
        return {
            'id': self.uuid_str(),
            'blocker_id': _uuid_to_str(self.blocker_id),
            'blocked_id': _uuid_to_str(self.blocked_id),
            'blocked_at': self.blocked_at.isoformat() if self.blocked_at else None,
            'reason': self.reason,
        }

