# TODO : (was code generated)
# # -*- coding: utf-8 -*-
# import unittest
# import os

# from flask import Flask
# from flask_testing import TestCase
# from flask_migrate import Migrate
# from repository.db import db
# from app import create_app
# from services.folderService import FolderService
# from services.videoService import VideoService

# class RouterTestCase(TestCase):
#     def create_app(self):
#         app = create_app("config.TestConfig")
#         app.json.compact = False
#         return app

#     def setUp(self):
#         self.folderService = FolderService()
#         self.videoService = VideoService()

#     def tearDown(self):
#         db.session.remove()
#         db.drop_all()

# class HealthRouterTestSuite(RouterTestCase):
#     def test_get_health(self):
#         response = self.client.get('/health')
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(response.get_json(), {"status": "ok"})

# class FolderRouterTestSuite(RouterTestCase):
#     def setUp(self):
#         super().setUp()
#         # Create test data
#         self.root_folder = self.folderService.create("root_test", None)
#         self.child_folder = self.folderService.create("child_test", self.root_folder)

#     def test_get_root_folders(self):
#         response = self.client.get('/folders')
#         self.assertEqual(response.status_code, 200)
#         data = response.get_json()
#         self.assertIn("Children", data)
#         self.assertEqual(len(data["Children"]), 1)  # Should have the root_test folder

#     def test_get_folder_by_id(self):
#         response = self.client.get(f'/folders/{self.root_folder.Id}')
#         self.assertEqual(response.status_code, 200)
#         data = response.get_json()
#         self.assertEqual(data["Name"], "root_test")
#         self.assertIn("Children", data)

#     def test_get_folder_invalid_id(self):
#         response = self.client.get('/folders/99999')
#         self.assertEqual(response.status_code, 404)

# class VideoInfoRouterTestSuite(RouterTestCase):
#     def setUp(self):
#         super().setUp()
#         # Create test data
#         self.test_folder = self.folderService.create("test_folder", None)
#         self.test_video = self.videoService.add("test_video.mp4", self.test_folder, 100, 1920, 1080, 25.0)

#     def test_get_video_info_valid(self):
#         response = self.client.get(f'/video/{self.test_video.Id}/info')
#         self.assertEqual(response.status_code, 200)
#         data = response.get_json()
#         self.assertEqual(data['Name'], 'test_video.mp4')

#     def test_get_video_info_invalid_id(self):
#         response = self.client.get('/video/99999/info')
#         self.assertEqual(response.status_code, 404)

