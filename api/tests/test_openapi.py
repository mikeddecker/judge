import unittest
from flask import Flask

from openapi import build_openapi_spec

class OpenApiExtractorTest(unittest.TestCase):
    def test_build_spec_basic(self):
        app = Flask(__name__)

        @app.route('/health')
        def health():
            """Health check"""
            return 'ok'

        @app.route('/video/<uuid:videoId>')
        def video(videoId):
            """Get video by id\nRetrieves video information for the provided UUID."""
            return ''

        with app.app_context():
            spec = build_openapi_spec(app)
            self.assertIn('paths', spec)
            self.assertIn('/video/{videoId}', spec['paths'])
            self.assertIn('get', spec['paths']['/video/{videoId}'])
            op = spec['paths']['/video/{videoId}']['get']
            self.assertEqual(op['summary'], 'Get video by id')

if __name__ == '__main__':
    unittest.main()

