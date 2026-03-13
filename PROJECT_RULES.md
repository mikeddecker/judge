
# Project Rules (AI Judge)

Purpose: short, actionable rules for contributors and maintainers so the project stays consistent and Docker-first - Makefile approach.

## Development Environment
- Use Python 3.12 (images and CI are configured for 3.12).
- Do not run `python app.py` locally — run services inside Docker. Use the repository's Docker Compose profiles for dev/test/prod.

## Docker and Running
- The canonical way to run services is via Docker Compose. Example (development) by using the Makefile:

```bash
docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up --build api web
```

- Ensure new services or endpoints are reachable inside the container and that `docker-compose` healthchecks pass before depending services are added.

## Code Style and Quality
- Always add type hints for public functions and methods where practical.
- Write clear docstrings for functions and classes; use the first line as a short summary.
- Use structured logging (project uses `structlog` by convention).
- Keep functions small and single-responsibility; prefer small, testable service/repo layers.

## Commits and Branches
- Use short, descriptive commit messages. Prefer present-tense summary and a short body when needed.
- Branch naming: `feature/<short-description>`, `fix/<short-description>`, `chore/<short-description>`, or `hotfix/<issue>`.

## Tests and CI
- Add unit tests for new logic under `api/tests` and ensure they run inside the `api-test` Docker service.
- CI must run tests inside Docker (mirror `api-test` service) to match environment and dependencies.

## Database & Migrations
- Use Alembic (Flask-Migrate) for schema changes. The migration script & migration can be done manually by the developer. See README of the API for instructions. These can be used inside the container during CI or deploy.
- Do not modify production schema without a migration and rollback plan.

## Documentation
- Keep `documentation/` up to date. Document architecture, domain models, API changes, and operational steps.
- The API exposes auto-generated docs at `/openapi.json` and a simple UI at `/docs`. Keep route docstrings up-to-date to improve generated docs.

## Security & Secrets
- Never commit secrets. Use `.env` files excluded from source control and Docker secrets for production.
- Default to privacy-by-design: opt-in features, minimal data retention, and explicit permission checks.

## Permissions & Auth
- Account permissions are documented in `documentation/IMPLEMENT_PERMISSIONS.md`. When changing auth/permissions, update docs and add migrations.

## Pull Requests & Reviews
- Open a PR for any non-trivial change. Include: summary, testing steps (Docker commands), any required migrations, and documentation updates.
- A PR should include at least one approving review before merge (two for major changes).

## Releases and Deploy
- Build and run releases using the Docker image and `docker-compose.prod.yaml` or CI/CD pipeline. Tag releases in git and attach migration notes.

## Small Notes
- IDs are UUIDs throughout the repo; prefer UUID types where applicable. (Code might still contain int type hints even though it's already UUID. Check for that)
- If you add or change API endpoints, ensure the OpenAPI generator and docs are correct.

## Makefile
- The repository includes a `Makefile` to simplify common Docker and development tasks. Prefer `make` targets over running `docker compose` manually when available.
- Common targets you may find or add:
	- `make dev` — start API and web in development mode (uses `docker-compose.dev.yaml`).
	- `make build` — build the Docker images.
	- `make up` / `make down` — bring services up or down via Docker Compose.
	- `make test` — run the test suite inside the `api-test` service.
	- `make migrate` — create/run database migrations (executes inside the API container).
	- `make backup` / `make restore` — backup or restore the database (runs inside container or uses provided scripts).
- Document any new `make` targets you add in `documentation/` and ensure they work inside the project's Docker profiles.

## Web
- Refer to other components or views and use the same style
- Use `<script setup>` for instance.

