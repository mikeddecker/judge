from flask import Flask, current_app
from flask_migrate import Migrate
from flask_testing import TestCase
from repository.models import Folder as FolderDB
import os
from dotenv import load_dotenv
load_dotenv()
from repository.db import db

# TODO : update tests, just a tryout to make it work
class MyTest(TestCase):
    def create_app(self):
        app = Flask(__name__)
        app.config.from_object("config.TestConfig")
        app.json.compact = False
        db.init_app(app)
        migrate = Migrate(app, current_app)
        migrate.init_app(app, db)
        with app.app_context():
            db.create_all()
        return app
    
    def test_server_is_up_and_running(self):
        assert True

    def test_server_count(self):
        for i in range(5):
            folder = FolderDB(name=f"competition{i}")
            db.session.add(folder)
        assert 5 == db.session.query(FolderDB.id).count()

    def test_server_count2(self):
        for i in range(5):
            folder = FolderDB(name=f"competition{i + 5}")
            db.session.add(folder)
        assert 5 == db.session.query(FolderDB.id).count()

    def test_contains_competition(self):
        K = 3
        folder = FolderDB(name=f"comp_{1}")
        db.session.add(folder)
        for i in range(K):
            folder = FolderDB(name=f"random_name_{i}", parentId=i+1)
            db.session.add(folder)
        for i in range(4):
            folder = FolderDB(name=f"other_name_{i}")
            db.session.add(folder)

        ls = db.session.query(FolderDB).filter(FolderDB.name == "random_name_2").all()
        print("LS", ls)
        assert 1 == len(ls)


    def setUp(self):
        print("setup calles.")
        pass

    def tearDown(self):
        db.session.remove()
        db.drop_all()