The account to which the account/ logs into.
Account has a type, such as Company/Organisation/Club/Team... (Whatever accounts want their organisation to be called.) Mainly functionality differs only for accounts and external services.

# Account = Account
(local account and/or SSO/OpenID connect)

RBAC: Possibility to switch to organisation view
# Account = Company/Organisation/Club/Team
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
- passwordHash
- salt
- lastLogin
- createdAt
- updatedAt
- mfaEnabled
- mfaCode
- mfaCodeExpires

# Permissions (issued by Admin/subscription)
![[AI-Judge.AccountPermissions]]
- ⏬ canTrainModels (💲)
- 

#TODO define other premium functions?