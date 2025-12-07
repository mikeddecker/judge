from flask_restful import Resource

class HealthRouter(Resource):
    def get(self):
        return {"status": "ok"}, 200

