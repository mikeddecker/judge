# Content
```table-of-contents
title: 
style: nestedList # TOC style (nestedList|nestedOrderedList|inlineFirstLevel)
minLevel: 0 # Include headings from the specified level
maxLevel: 0 # Include headings up to the specified level
include: 
exclude: 
includeLinks: true # Make headings clickable
hideWhenEmpty: false # Hide TOC if no headings are found
debugInConsole: false # Print debug info in Obsidian console
```
# Summary

![[AI-Judge.excerpt.Security-summary]]

# 📋 Backlog

![[AI-Judge.excerpt.Security-features-backlog]]

# ☑ Password (requirements)

(local user)
Current requirements
- Minimum 12 characters (defined by `.env`)
- No capitol, special char etc check yet

Password security
- Service layer with PBKDF2-SHA256 password hashing and email support
	- `hashlib.pbkdf2_hmac('sha256', ...`
- Salt
# ⏸️ MFA 🔢

Why? look up on the internet

Options: Only Email 6 digit TOTP (Time-based, One-Time Password)
No email provider yet. I can't connect to outlook with an app password.
I can't connect (personal) outlook with an app password. (Invalid)
Integration on stand-by.

Other forms of MFA
- 💲SMS
- 💲Push notification
- 💲💲 Physical keys

# 🛸 External authentication

## OpenID connect 
For users #TODO
(https://openid.net/developers/how-connect-works/)

## OAuth 2.0
For applications #TODO

# 🔒🚪Authorization : RBAC

Deciding who gets to do what.
RBAC = Role Based Access Control




# References

- [[AI-Judge.Login-security.2026-02-07.Copilot-summary]]
- [[AI-Judge.Login-security.2026-02-07-chatGPT-explanation]]

