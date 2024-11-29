from flask import request, jsonify
from flask_restful import Resource

from repository.models import Folder, Video, db
import os
from dotenv import load_dotenv
load_dotenv()

class FolderRouter(Resource):
    # read
    def get(self, user_id=None):
        if user_id:
            user = Folder.query.get(user_id)
            if not user:
                return {'error': 'User not found'}, 404
            return {'user': user.to_dict()}, 200
        else:
            users = Folder.query.all()
            return {'users': [user.to_dict() for user in users]}, 200

    # create
    def post(self):
        data = request.get_json()
        path = data.get('path')
        if not path:
            return {'error': 'Path required'}, 400
        new_user = Folder(path=path)
        db.session.add(new_user)
        db.session.commit()
        return {'message': 'User added successfully'}, 201

    # update
    def put(self, user_id):
        user = Folder.query.get(user_id)
        if not user:
            return {'error': 'User not found'}, 404
        data = request.get_json()
        path = data.get('path')
        if not path:
            return {'error': 'Path required'}, 400
        user.path = path
        db.session.commit()
        return {'message': 'User updated successfully'}, 200

    # delete
    def delete(self, user_id):
        user = Folder.query.get(user_id)
        if not user:
            return {'error': 'User not found'}, 404
        db.session.delete(user)
        db.session.commit()
        return {'message': 'User deleted successfully'}, 200
