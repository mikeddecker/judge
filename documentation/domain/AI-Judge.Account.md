The account to which the user/ logs into. 
Account has a type, such as Company/Organisation/Club/Team... (Whatever users want their organisation to be called.) Mainly functionality differs only for users and external services.

# Account = User
(local user and/or SSO/OpenID connect)

RBAC: Possibility to switch to organisation view
# Account = Company/Organisation/Club/Team
e.g. Gymfed, NextJump, IJRU, AMJRF...
e.g. Sipiro, KangarooKids, Siluskip...
(local user and/or SSO/OpenID connect)

RBAC: Can assign users who can view the application as this organisation
# Account external service
(requires OAuth 2.0)

Import/export data from/to other services

# Account = User (admin)
(local user and/or SSO/OpenID connect)

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
- ⏬ canTrainModels (💲)

#TODO define other premium functions?