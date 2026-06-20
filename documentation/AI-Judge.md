# 🏁 Goal
Assisting judges in scoring jump routines
- Requiring less judges a team/club has to offer on competitions
- Decrease human error during competition judging
- Increase transparency of judging
- (Benefit: engage the public more, by explained scores)

# 📋 Content
```table-of-contents
```
# 💻 Project stage: development

What does this mean? Most aspects are still in the development stage. Even though a lot is built, much more features are waiting in line to be developed.

# 🧭 High level overview

- Browse videos
- Label frames (localization)
- Segment videos
- Recognize (skill) elements in te segment
- Organisations (multi-tenancy)
	- Org members
	- Individual members, representatives
	- Upload videos (set them public/private)
		- On the folder or video
		- Permission an video wins
	- Give permissions to other users (friends?), groups...
- Browse videos
	- In folder mode (like browsing a directory)
	- In tag mode (where 'folders' represent tags)
	- (Forsee pages)
- Label video frames
- Label video (fragments)/segments
- Configure layers & layercompositions
- See statistics, train results

This repository contains a working Flask REST API (`/api`) and a front-end (`/web`) plus a computer-vision training and prediction stack (`/computervision`). The API is the canonical interface for creating, reading, updating and deleting videos, folders, labels, accounts and model jobs. The web application is a lightweight client that uses session-based authentication and the API routes.

# 🔒 Security (by design?)

![[AI-Judge.excerpt.Security-summary]]

More details in [[AI-Judge.Security-by-design]]
# 🕶 Privacy by design

![[AI-Judge.Privacy-by-design-summary]]

Key privacy rules implemented in code:

- Videos are private by default (database field `Videos.private` default true). Access must be granted explicitly via owner grants; folder-level grants are supported conceptually but video-level rules prevail.
- Server-side sessions are used for authentication (`/auth/login` sets `session['account_id']`). See `api/routers/accountRouter.py` for exact flow.
- Blocking rules should always override grants (design documented in `documentation/features/AI-Judge.Permissions.md`).

# ✴ Availability

See [[AI-Judge.Multi-Region]] for detailed multi-region deployment architecture across Belgium (EU) and USA.

Current plan concepts:
- Primary instance in Belgium (EU)
- Passive/replica instance in USA
- Manual DNS failover (or auto-failover via cloud LB in future)
- Database replication (primary → replica)
- Video storage sync (S3 or rsync)
- Database backup automation (6-hourly encrypted backups)

# 🏬 Backup
![[AI-Judge.Backup]]

# 👁‍🗨 Monitoring
None yet, really low prio.
- (Email/SMS/...) notification if service down?
- if service peaking on usage?
- if model trained?
Logs? very little to none -> GDPR?
# 🎡 CI/CD
...

# 🔗 Integration into NextJump.app
Integration into nextjump (subdomain on nextjump.app?)
    - Do I register [nextjump.be](http://nextjump.be) as well? (about 10 euro/year)
    - For more simple load balancing in the eu
Public facing side
vs
Competition/IJRU/Gymfed... facing side (NGB's) - 'current' focus

# 👤 Business continuity?
None yet
Also no hard requirement, unless it is adopted into judging panels.
Because ...
# 🧯Disaster Plan Recovery
See [[AI-Judge.Disaster-Recovery]] for detailed operational runbooks, incident response procedures, and recovery steps for all critical failure scenarios. Includes:
- Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)
- Failover procedures (database, API, video storage)
- Data restoration from backups
- Quarterly DR drills
- On-call escalation paths

# 💬 Discussion points
Computervision
    - Training possible on two sides (results in DB)
    - I think you can focus on this part a bit more?
    - I think there is a bug in the numeric label train/test labels
    - Maybe review it together?
    - Reads out the database -> provided labels.
    - Do we directly integrate/migrate this towards NextJump?
    - Ideas I had was instead of pre-filtering based on full json string -> filtering on layer value occurrence. If then a label is a null tensor (or a fully masked one > skip)
    - Next idea, calculate acc/f1 based on tags/output_heads instead of full layers -> which gives more info about accuracy of crosses depending on the rotation.
    - Review how to label wraps/DD-transitions/snappers/skillsegments having multiple rope rotations, but which are not multiples.
    - While we're at it, review the current setup of the layercomposition. It might be able to be slightly more easier.
- API -> for querying/posting/deleting videos, folders, accounts, labels.
- API -> for querying/posting/deleting videos, folders, accounts, labels and jobs.
        - The API is located in `api/` and exposes RESTful resources via `flask_restful`.
        - Notable routes (file: `api/app.py`):
            - `POST /auth/login`, `POST /auth/register`, `GET /auth/me`, `POST /auth/logout` — session-based auth.
            - `GET/POST /video/<uuid:videoId>` — download/update video metadata (update requires authentication).
            - `GET /video/<uuid:videoId>/info` — get video metadata as JSON.
            - `GET /video/<uuid:videoId>/predictions` — retrieve aggregated predictions (skills + boxes).
            - `GET/POST /video/<uuid:videoId>/image` — thumbnail retrieval and (disabled) image extraction endpoint.
            - `GET/frames`, `POST /skill/<uuid:videoId>`, `POST /skillcompleted/<uuid:videoId>` — labeling & skill endpoints.
            - `POST /job/retrain` and `POST /job` — ML job launch and retrain hooks handled by `services/jobService.py`.

        - Router responsibilities: extract authentication identity (via Flask `session`) and input validation.
        - Service responsibilities: authorization & business rules (see `api/services/videoService.py`).

        - Repository responsibilities: database access and mapping to domain objects (see `api/repository/*`).
- WEB app, Idk if you have one right now?
    - Can remain mostly as-is -> expand on it (dev ideas...)
    - 'Quick' setup: old pc -> only web + api
        - Then develop the database/file sync? (I have 2 external SSD's of 1TB of video's now)
        - When my pc is online -> sync video's/labels/database

# � Roadmap

See [[AI-Judge.Roadmap]] for the comprehensive development roadmap with:
- **Phase 1**: Foundation & Security (permissions, MFA, type hints, CI/CD)
- **Phase 2**: Multi-Region Infrastructure (database replication, load balancing, secrets)
- **Phase 3**: Observability (monitoring, logging, backups, disaster recovery)
- **Phase 4**: Advanced Features (OpenID, OAuth, admin panel, geo-replication)

Integrated with operational procedures in [[AI-Judge.Disaster-Recovery]] and infrastructure details in [[AI-Judge.Multi-Region]].

---

**Developer notes & useful references (quick lookups)**

- Authentication flow and session usage: see [api/routers/accountRouter.py](api/routers/accountRouter.py#L1).
- Video endpoints and update authorization: see [api/routers/videoRouter.py](api/routers/videoRouter.py#L1) and [api/services/videoService.py](api/services/videoService.py#L1). The service enforces field-level permissions and currently blocks changing the `training` flag for non-admins.
- Database models and UUID handling: `api/repository/models.py` uses a 16-byte binary `UUIDType`; a generated SQL helper adds human-readable `uuid_str` columns via `repository/calculated_columns.sql`.
- Permissions: design documented in [documentation/features/AI-Judge.Permissions.md](documentation/features/AI-Judge.Permissions.md#Overview).
- Computer vision training and inference code: `computervision/` contains trainers (`Trainer.py`), predictor (`Predictor.py`), the model skeletons, and recipes (`recipes.json`). Jobs use `services/jobService.py` to schedule/launch.

**How to run locally (quick)**

1. Set environment variables in `api/.env` (or export in shell) for MySQL and secrets: `MYSQL_HOST`, `MYSQL_USERNAME`, `MYSQL_ROOT_PASSWORD`, `MYSQL_DATABASE`, `STORAGE_DIR_VIDEOS`, `STORAGE_DIR_GENERATED_DATA`, `SECRET_KEY`.

2. Start the API:
```bash
export FLASK_APP=api.app
export FLASK_ENV=development
python api/app.py
```

3. Access the web client (if running) which communicates with the API via the same host/port. Browser handles Flask session cookie.

4. For ML jobs, see `computervision/README.md` — typically run in a separate process or container; jobs can be launched via the `/job` API endpoints which will enqueue or run training/inference logic.

**Testing & verification**

- Unit tests are present under `api/tests/` — run them from the `api` folder:
```bash
cd api
pytest -q
```

**Future work (from roadmap and current code gaps)**

- Implement DB-backed `AccountCapability`, `AccessGrant`, and `AccountBlock` tables and wire them into `AccountRepo` and service-level auth checks. See `documentation/features/AI-Judge.Permissions.md` for schema suggestions.
- Add admin/capability checks to permit changing `training` flag and other sensitive operations.
- Add monitoring/alerting and more extensive logging (GDPR-aware) for production.
- Harden session cookies and consider switching to JWT for stateless API clients.

If you want I can now scaffold the DB models and initial alembic migration for the permissions tables, then wire permission checks to allow `training` flag changes only for accounts with the appropriate capability.



```bash
judge-infra/          ← as described before, no changes
judge-db-api/
  .github/workflows/
    ci.yml            ← test + lint on PR
    publish-package.yml
    deploy.yml        ← auto on merge to main
  packages/
    judge-db-models/  ← published, installed by judge-cv
  migrations/         ← Flyway SQL files
  api/                ← Flask app
  AGENTS.md
judge-web/
  .github/workflows/
    ci.yml
    deploy.yml
  src/                ← your existing Vue code
  docs/               ← VitePress
  AGENTS.md
judge-cv/
  .github/workflows/
    ci.yml
    build.yml         ← builds GPU Docker image
  computervision/     ← your existing CV code
  AGENTS.md
```
  