# Account Type = user
(local account and/or SSO/OpenID connect)

RBAC: Possibility to switch to organisation view (of which it is a representative)
# Account Type = Company/Organisation/Club/Team
e.g. Gymfed, NextJump, IJRU, AMJRF...
e.g. Sipiro, KangarooKids, Siluskip...
(local account and/or SSO/OpenID connect)

RBAC: Can assign accounts who can view the application as this organisation
# Account external service
(requires OAuth 2.0)

Import/export data from/to other services

# Account = Account (admin)
(local account and/or SSO/OpenID connect)

RBAC: Possibility to switch to views of another account
# Account - Properties:
- id
- email
- firstName
- lastName
- passwordHash (PBKDF2-hashed with a per-account salt in `AccountService.hash_password()`)
- salt
- lastLogin
- createdAt
- updatedAt
- mfaEnabled
- mfaCode
- mfaCodeExpires
- type (admin, user, organisation, team, club, group)

Fields & behavior
- MFA flow: `AccountService.login()` will require MFA if `mfaEnabled` is true; `AccountMFAVerifyRouter` completes the flow.
- Sessions: successful login stores `session['account_id']` and `session['email']` (server-side Flask session).

Repo helpers
- `AccountRepo.get_account_by_email(email)` and `get_account_by_id(id)` return domain `Account` objects.
- `AccountRepo.update_lastLogin(id)` updates `lastLogin` timestamp.

⚠ Notes & recommendations
- Do not expose `passwordHash` or `salt` in API responses. `MapToDomain.map_account()` returns a domain object that omits these fields for JSON responses.
- Consider moving sensitive audit fields (password resets, MFA attempts) into a separate audit table for compliance.

# AccountTypes
admin, user (individual), group
In essense: 
- Groups can contain


# Permissions (issued by Admin/subscription)
![[AI-Judge.AccountPermissions]]
- ⏬ canTrainModels (💲)
- 

#TODO define other premium functions?

