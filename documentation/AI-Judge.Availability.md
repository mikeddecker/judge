The core principles of availability for the AI-Judge.

- down/incidents = down? -> best effort principle?
- sync between servers & videos -> uuid makes this better possible
	- astronomically small chance that uuid collides
	- check that data matches between video (name)
	- if not -> change the one with the latest createdAt timestamp.

# Dev instance (running locally)

# Test instance (like prod, but test)

# Prod active instance EU&US?

# Prod passive instance EU&US?
fallback system within the EU
- Manual DNS failover -> change A record to passive instance?
- Auto switch to passive instance?

Later: preferably in the cloud/in a datacenter.
