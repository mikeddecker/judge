from flask import jsonify, current_app, Response
import json
import traceback
import re

def _flask_rule_to_openapi_path(rule):
    # Convert Flask-style paths like '/video/<uuid:videoId>' to OpenAPI '/video/{videoId}'
    path = re.sub(r"<[^:>]+:([^>]+)>", r"{\1}", rule)
    path = re.sub(r"<([^>]+)>", r"{\1}", path)
    return path

def _param_type_from_converter(converter):
    # Map common Flask converters to OpenAPI types
    mapping = {
        'int': ('integer', 'int32'),
        'float': ('number', 'float'),
        'path': ('string', None),
        'string': ('string', None),
        'uuid': ('string', None),
    }
    return mapping.get(converter, ('string', None))

def build_openapi_spec(app):
    spec = {
        'openapi': '3.0.0',
        'info': {
            'title': app.config.get('OPENAPI_TITLE', 'AI Judge API'),
            'version': app.config.get('OPENAPI_VERSION', '0.1.0'),
            'description': app.config.get('OPENAPI_DESCRIPTION', 'Auto-generated API documentation'),
        },
        'paths': {},
    }

    # Iterate over url rules
    for rule in app.url_map.iter_rules():
        # Skip static endpoints and the openapi endpoints themselves
        if rule.endpoint.startswith('static'):
            continue
        if rule.rule.startswith('/openapi') or rule.rule.startswith('/docs'):
            continue

        path = _flask_rule_to_openapi_path(rule.rule)
        methods = [m for m in rule.methods if m in ('GET', 'POST', 'PUT', 'PATCH', 'DELETE')]
        if not methods:
            continue

        path_item = spec['paths'].setdefault(path, {})

        for method in methods:
            view_func = app.view_functions.get(rule.endpoint)
            summary = None
            description = None
            if view_func and view_func.__doc__:
                # first line summary, rest description
                doc = view_func.__doc__.strip()
                parts = doc.split('\n', 1)
                summary = parts[0].strip()
                if len(parts) > 1:
                    description = parts[1].strip()

            # Build parameters from rule.arguments
            params = []
            for arg in rule.arguments:
                # try to infer converter from the rule text
                m = re.search(r"<([^:>]+):%s>" % re.escape(arg), rule.rule)
                converter = None
                if m:
                    converter = m.group(1)
                elif f"<{arg}>" in rule.rule:
                    converter = 'string'

                p_type, p_format = _param_type_from_converter(converter)
                param = {
                    'name': arg,
                    'in': 'path',
                    'required': True,
                    'schema': {'type': p_type},
                }
                if p_format:
                    param['schema']['format'] = p_format
                params.append(param)

            op = {
                'summary': summary or f'{method} {path}',
                'description': description or '',
                'responses': {
                    '200': {
                        'description': 'Successful response'
                    }
                }
            }
            if params:
                op['parameters'] = params

            path_item[method.lower()] = op

    return spec

def attach_openapi_endpoints(app):
    @app.route('/openapi.json')
    def openapi_json():
        try:
            spec = build_openapi_spec(current_app)
            body = json.dumps(spec)
            return Response(body, mimetype='application/json')
        except Exception as e:
            # Log exception server-side for debugging
            traceback.print_exc()
            # Return a minimal, valid OpenAPI document so Swagger UI shows a readable error
            fallback = {
                'openapi': '3.0.0',
                'info': {
                    'title': current_app.config.get('OPENAPI_TITLE', 'AI Judge API'),
                    'version': current_app.config.get('OPENAPI_VERSION', '0.0.0'),
                    'description': f'Failed to build OpenAPI spec: {str(e)}',
                },
                'paths': {},
            }
            return Response(json.dumps(fallback), mimetype='application/json')

    @app.route('/docs')
    def docs_page():
        # Simple Swagger UI HTML that loads the generated /openapi.json
        html = '''<!doctype html>
            <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>API Docs</title>
                <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@4/swagger-ui.css" />
            </head>
            <body>
                <div id="swagger-ui"></div>
                <script src="https://unpkg.com/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
                <script>
                const ui = SwaggerUIBundle({
                    url: '/openapi.json',
                    dom_id: '#swagger-ui',
                })
                </script>
            </body>
            </html>'''
        return Response(html, mimetype='text/html')

