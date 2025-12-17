from flask import current_app
from flask_restful import Resource

class HealthRouter(Resource):
    def get(self):
        response = {"status": "ok"}
        if current_app.config.get('TESTING', False):
            response["test_mode"] = True
            response["database"] = "test"
        return response, 200

