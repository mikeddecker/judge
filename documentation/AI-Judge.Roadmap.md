

![[Pasted image 20260406131138.png]]

![[Pasted image 20260406132306.png]]

Images above in words:
Volume fallback: 
- Production (old laptop) -> fail -> needs to be there (external SSD) 
- Production computervision (new laptop) -> only when judge-cv service is spinned on 
  (a command to spin up, a command to spin down) 
- Dev -> Local (dev laptop) -> No external ssd needed. test -> on old laptop -> second instance -> test.domain.com

**`judge-infra`** (start here, day 1) It contains:
- Docker Swarm stack files
- Keycloak configuration (realm export, client configs, roles/groups), 
- Traefik as reverse proxy with automatic HTTPS via Let's Encrypt
- PostgreSQL with tablespace init scripts, 
- Redis - Task queue 
- Portainer for visual swarm management
- Volume fallback logic. No application code lives here — just infrastructure as code. Private repo.

**`judge-platform`** (start second) Contains three services that deploy together: the FastAPI backend, the Vue/Nuxt frontend, and the docs site. The docs site is a dead-simple service (VitePress or Docusaurus) that watches a `/docs` folder of markdown files and builds them into a navigable website — deployed automatically at `docs.yourdomain.com`. This is the repo you'll touch daily. Public or private — your call.

**`judge-cv`** (start third) The computer vision worker. GPU-dependent, separate Python environment, CUDA base image. Communicates with the platform via the task queue (Redis/Celery or ARQ) and stores model metadata in the shared PostgreSQL. Private repo because it likely contains proprietary training data paths and model weights.










A basic roadmap to list priorities
```table-of-contents
```
# Prepare multi user labeling
Prepare for multi user labeling.
This includes:
- Authentication
- Authorisation (RBAC)
	- [[AI-Judge.Privacy-by-design]]
	- Groups, organisations...
		- including admin group
- Security ([[AI-Judge.Security-by-design]])


# Multi-user labeling 
- Of your own data
- Of public data
- Other permissions finegraining later?
- Train on all
- Validate on all
- Validate score visible for own data
## Host in US or EU



# Finer data control / permissions
- On folders (train/test)
- On specific videos (train/test)
- On folders (see/edit/label/block)
- On videos (see/edit/label/block)
- On accounts (see/edit/label/block)
- On

## Host in US and EU
- with data sync
- increase/safeguard availability


# Future
- Sport agnostic
	- Public Label presets (jump rope, acro, tumbling, )
	- Custom layers (crosses, multiples, body rotations, hands, feet...)
		- Ability to make layers public
		- Ability to share layers
- Upload of videos
	- One by one
	- Using SSD -> directory discovery ✅
	- Using other means, e.g. API calls.