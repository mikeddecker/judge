from flask import request, Response
from flask_restful import Resource
from services.folderService import FolderService
from services.videoService import VideoService
from services.jobService import JobService
from services.storageService import StorageService
from services.tagService import TagService
from helpers.ValueHelper import ValueHelper
from uuid import UUID

class TagRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        self.jobService = JobService()
        self.tagService = TagService()
        super().__init__(**kwargs)

    def get(self):
        return [t.to_dict() for t in self.tagService.get_tags()], 200

    def post(self):
        data = request.get_json()
        name = data.get('name')
        group = data.get('group')

        if not group:
            return 'No group given', 400
        try:
            ValueHelper.check_raise_string_only_abc123(name)
            return self.tagService.add(name, group).to_dict(), 200
        except ValueError as err:
            return str(err), 400

    def put(self):
        data = request.get_json()
        id = UUID(data.get('id'))
        name = data.get('name')
        keywords = data.get('keywords')
        group = data.get('group')

        ValueHelper.check_raise_uuid(id)
        if not self.tagService.has_tag(id):
            raise ValueError(f"Tag with id {id} does not exist")

        if name is None and keywords is None:
            if not self.tagService.has_tag_group(group):
                raise ValueError(f"TagGroup {group} does not exist")
            else:
                if group is None:
                    ValueHelper.check_raise_string_only_abc123(group)
                self.tagService.update_tag_group(tag_id=id, group_name=group)
        else:
            if name is None:
                ValueHelper.check_raise_string_keywords_comma_separated(keywords)
                self.tagService.update_tag_keywords(tag_id=id, keywords=keywords)
            else:
                ValueHelper.check_raise_string_only_abc123(name)
                self.tagService.update_tag_name(tag_id=id, new_name=name)

class TagGroupRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        self.jobService = JobService()
        self.tagService = TagService()
        super().__init__(**kwargs)

    def get(self):
        return [t.to_dict() for t in self.tagService.get_tag_groups()], 200

    def post(self):
        data = request.get_json()
        name = data.get('name')
        ValueHelper.check_raise_string_only_abc123(name)
        return self.tagService.add_group(name).to_dict(), 200

class TagDiscoveryRouter(Resource):
    def __init__(self, **kwargs):
        self.storageService = StorageService()
        super().__init__(**kwargs)
    
    def get(self):
        try:
            return self.storageService.discover_tags(), 200
        except Exception as ve:
            return Response(str(ve), status=500)

