from domain.tag import Tag
from domain.tagGroup import TagGroup
from domain.videoinfo import VideoInfo
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from repository.jobRepo import JobRepository
from repository.tagRepo import TagRepository
from typing import List
from helpers.ValueHelper import ValueHelper
from uuid import UUID

class TagService:
    """Provides the video information of videos"""
    def __init__(self):
        self.VideoRepo = VideoRepository(db=db)
        self.FolderRepo = FolderRepository(db=db)
        self.JobRepo = JobRepository(db=db)
        self.TagRepo = TagRepository(db=db)

    def get_tags(self) -> List[Tag]:
        """Return all tags from the DB"""
        return self.TagRepo.get_tags()

    def get_tag_groups(self) -> List[TagGroup]:
        """Return all tag groups from the DB"""
        return self.TagRepo.get_tag_groups()

    def add(self, name: str, group: str | None) -> Tag:
        """Let's just say double names are possible, only not in the same group"""
        ValueHelper.check_raise_string_only_abc123(name)
        if group and not self.TagRepo.has_group(group):
            raise ValueError(f"Group {group} does not exist")
        if self.TagRepo.has_tag_within_group(name, group):
            raise ValueError(f"Tag {name} already exists in group {group}")

        return self.TagRepo.add(name, group)

    def add_group(self, name) -> TagGroup:
        ValueHelper.check_raise_string_only_abc123(name)
        if self.TagRepo.has_group(name):
            raise ValueError(f"Groupname {name} already exists")
        return self.TagRepo.add_group(name)

    def has_tag(self, id: UUID) -> bool:
        ValueHelper.check_raise_uuid(id)
        return self.TagRepo.has_tag(id)

    def has_tag_group(self, name: str) -> bool:
        if name is None:
            return True
        ValueHelper.check_raise_string_only_abc123(name)
        return self.TagRepo.has_group(name)

    def update_tag_group(self, tag_id: UUID, group_name: str | None):
        ValueHelper.check_raise_uuid(tag_id)
        assert self.TagRepo.has_group(group_name), f"TagGroup {group_name} does not exist"
        return self.TagRepo.update_tag_group(tag_id, group_name)

    def update_tag_name(self, tag_id: UUID, new_name: str):
        ValueHelper.check_raise_uuid(tag_id)
        ValueHelper.check_raise_string_only_abc123(new_name)
        assert self.TagRepo.has_tag(tag_id), f"Tag with id {tag_id} does not exist"
        return self.TagRepo.update_tag_name(tag_id, new_name)

    def update_tag_keywords(self, tag_id: UUID, keywords: str):
        ValueHelper.check_raise_uuid(tag_id)
        ValueHelper.check_raise_string_keywords_comma_separated(keywords)
        assert self.TagRepo.has_tag(tag_id), f"Tag with id {tag_id} does not exist"
        return self.TagRepo.update_tag_keywords(tag_id, keywords)

    def link_video_to_tag(self, tag_id: UUID, video_id: UUID):
        ValueHelper.check_raise_uuid(tag_id)
        ValueHelper.check_raise_uuid(video_id)

