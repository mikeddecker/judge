Admins can do anything and see the application simulated as another person. It will be a non admin user - e.g. specific user or a tenant owner/member/representative + option to go back to admin mode.

## How to make an admin
- Yes. For the new global storage-admin endpoints, assign realm role judge-platform-admin (or admin) to the user in the same realm used by the app.
- Steps:

1. Keycloak Admin Console -> Realm -> Realm roles -> create role judge-platform-admin (if missing).
2. Users -> select user -> Role mapping -> assign judge-platform-admin.
3. User logs out and logs in again to refresh token roles.