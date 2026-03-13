# API Reference (quick)

This file summarizes main API endpoints exposed by `api/app.py`. All endpoints are relative to the API root (default `http://localhost:5555`).

Authentication
- `POST /auth/login` — body: `{ "email": "..", "password": ".." }`. On success sets a server-side session cookie and returns account info (or `requires_mfa`).
- `POST /auth/register` — create account.
- `GET /auth/me` — returns `{'success': True, 'account': ...}` when session present; otherwise `success: False`.
- `POST /auth/logout` — clears session.

Video endpoints
- `GET /video/<uuid:videoId>` — returns raw video bytes (no auth required if public, otherwise depends on service grants).
- `POST /video/<uuid:videoId>` — update video metadata (requires authentication; body: subset of `Videos` fields, e.g. `{ "name": "x.mp4", "private": false }`). Service enforces allowed fields and training flag protection.
- `GET /video/<uuid:videoId>/info` — JSON metadata for video (includes frames, skills, tags).
- `GET /video/<uuid:videoId>/predictions` — aggregated predictions (skills + boxes) produced by ML jobs.
- `GET /video/<uuid:videoId>/image` — thumbnail image bytes.

Frame & labeling
- `GET/POST /video/<uuid:videoId>/frameNr/<int:frameNr>` — frame-specific operations; POST for frame image actions (some features not implemented).
- `GET /frameLabelTypes` — list available label types.

Skills
- `POST /skill/<uuid:videoId>` — add skill; body: `{ "frameStart": <int>, "frameEnd": <int>, "skillinfo": { ... } }`.
- `POST /skillcompleted/<uuid:videoId>` — mark skill labeling completed for a video.

Jobs
- `POST /job` — launch a job (train/predict). Body: `{ "type": "TRAIN", "step": "FULL", "job_arguments": { ... } }`.
- `POST /job/retrain` — retrain helper endpoint.

Tags, Layers, Results, Stats
- `GET/POST /tags`, `GET /tagGroups` — tag management.
- `GET/POST /layers`, `GET /layers/types`, `GET /layercompositions` — layer composition and values.
- `GET /results`, `GET /stats` — retrieve aggregated metrics and training results.

Examples (curl)
```bash
# login (saves session in cookie jar)
curl -c cookies.txt -H "Content-Type: application/json" -d '{"email":"u@example.com","password":"pw"}' http://localhost:5555/auth/login

# update video metadata (send cookie)
curl -b cookies.txt -H "Content-Type: application/json" -X POST -d '{"name":"video.mp4","private":false}' http://localhost:5555/video/<video-uuid>
```

Notes
- The router authenticates via Flask session. For automated clients prefer cookie handling or implement JWT gateway.
- Service-layer enforces authorization; repository code does not check permissions.


