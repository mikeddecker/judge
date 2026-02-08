# Authentication System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (Vue 3 + Tailwind)                 │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ LoginView    │  │ RegisterView │  │ ForgotPasswordView│   │
│  └──────────────┘  └──────────────┘  └───────────────────┘   │
│         │                │                      │               │
│         └────────────────┴──────────────────────┘               │
│                         │                                       │
│                  ┌──────▼──────┐                               │
│                  │  authStore  │◄─┐                            │
│                  │  (Pinia)    │  │                            │
│                  └──────┬──────┘  │                            │
│                         │         │                            │
│                    Router Guards  │                            │
│                         │         │                            │
└─────────────────────────┼─────────┼────────────────────────────┘
                          │         │
                    HTTP Requests   │
                          │         │
┌─────────────────────────▼─────────┼────────────────────────────┐
│                     API (Flask)    │                            │
│                                    │                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  User Routers                            │  │
│  │  /auth/register  /auth/login  /auth/mfa/verify etc.    │  │
│  └──────────────────┬─────────────────────────────────────┘  │
│                     │                                         │
│  ┌──────────────────▼─────────────────────────────────────┐  │
│  │              UserService (Business Logic)              │  │
│  │  • Hash passwords (PBKDF2-SHA256)                      │  │
│  │  • Generate/verify MFA codes                           │  │
│  │  • Send emails                                         │  │
│  │  • Validate inputs                                     │  │
│  └──────────────────┬─────────────────────────────────────┘  │
│                     │                                         │
│  ┌──────────────────▼─────────────────────────────────────┐  │
│  │          UserRepository (Data Access)                  │  │
│  │  • Create/read/update user                             │  │
│  │  • Manage MFA codes                                    │  │
│  │  • Query by email/ID                                   │  │
│  └──────────────────┬─────────────────────────────────────┘  │
│                     │                                         │
└─────────────────────┼─────────────────────────────────────────┘
                      │
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│                  Database (MySQL)                             │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Users Table                                           │  │
│  │  ├─ id (auto-increment)                               │  │
│  │  ├─ email (unique)                                    │  │
│  │  ├─ firstName, lastName                            │  │
│  │  ├─ passwordHash, salt                              │  │
│  │  ├─ lastLogin                                        │  │
│  │  ├─ createdAt, updatedAt                           │  │
│  │  ├─ mfaEnabled                                      │  │
│  │  └─ mfaCode, mfaCodeExpires                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Session Storage (Filesystem)                          │  │
│  │  └─ /flask_sessions/                                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Email Service (SMTP)                                  │  │
│  │  ├─ MFA code emails                                   │  │
│  │  └─ Password reset emails                             │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Authentication Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        LOGIN FLOW                                │
└──────────────────────────────────────────────────────────────────┘

1. USER INPUT
   ┌─────────────────────┐
   │ Enter email/password │
   └────────┬────────────┘
            │
            ▼
2. CLIENT VALIDATION
   ┌─────────────────────────────────────────┐
   │ • Check email format                    │
   │ • Check password not empty              │
   │ • Submit form                           │
   └────────┬────────────────────────────────┘
            │
            ▼
3. API VALIDATION & AUTH
   ┌─────────────────────────────────────────┐
   │ POST /auth/login                        │
   │ • Find user by email                    │
   │ • Hash submitted password               │
   │ • Compare hashes                        │
   └────────┬────────────────────────────────┘
            │
            ├─── Password Invalid? ──► Return 401 Error
            │
            ▼
4. UPDATE USER
   ┌─────────────────────────────────────────┐
   │ • Set lastLogin = now()               │
   │ • Update updatedAt timestamp           │
   └────────┬────────────────────────────────┘
            │
            ▼
5. CHECK MFA STATUS
   ┌─────────────────────────────────────────┐
   │ Is mfaEnabled = TRUE?                  │
   └─┬──────────────────────────────────┬────┘
     │                                  │
     ▼ YES                              ▼ NO
  6a. GENERATE MFA          6b. CREATE SESSION
  ┌──────────────────┐      ┌──────────────────┐
  │ • Gen 6-digit    │      │ • Save user_id   │
  │ • Set expires    │      │ • in session     │
  │ • Send email     │      │ • Return user    │
  │ • Return user_id │      │ object           │
  └─────────┬────────┘      └──────────┬───────┘
            │                          │
            ▼                          ▼
       7a. FRONTEND              7b. FRONTEND
       Shows MFA input           Redirects home
            │                          │
            │                          ▼
            │                   ┌────────────────┐
            │                   │ Access protected│
            │                   │ routes         │
            │                   └────────────────┘
            │
            ▼
       7c. USER ENTERS CODE
       ┌──────────────────┐
       │ Input 6 digits   │
       │ POST /auth/mfa/  │
       │      verify      │
       └────────┬─────────┘
                │
                ▼
          8. VERIFY CODE
          ┌──────────────────────────┐
          │ • Find user by user_id   │
          │ • Check code matches     │
          │ • Check not expired      │
          └────────┬─────────────────┘
                   │
                   ├─── Code Invalid? ──► Return 401 Error
                   │
                   ▼
          9. CREATE SESSION
          ┌──────────────────┐
          │ • Save user_id   │
          │ • Return user    │
          └────────┬─────────┘
                   │
                   ▼
          10. FRONTEND
          ┌──────────────────┐
          │ Redirects to home│
          └──────────────────┘
```

## Password Reset Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                   PASSWORD RESET FLOW                            │
└──────────────────────────────────────────────────────────────────┘

1. REQUEST RESET
   ┌─────────────────────────────────────────┐
   │ POST /auth/forgot-password              │
   │ Email: user@example.com                 │
   └────────┬────────────────────────────────┘
            │
            ▼
2. FIND USER
   ┌─────────────────────────────────────────┐
   │ Look up user by email                   │
   │ (Don't reveal if exists - security)     │
   └────────┬────────────────────────────────┘
            │
            ├─── User not found? ──► Return generic success
            │
            ▼
3. GENERATE RESET CODE
   ┌─────────────────────────────────────────┐
   │ • Generate random token                 │
   │ • Set expires = now + 1 hour            │
   │ • Store in user record                  │
   └────────┬────────────────────────────────┘
            │
            ▼
4. SEND EMAIL
   ┌─────────────────────────────────────────┐
   │ • Compose reset email                   │
   │ • Include reset code/link                │
   │ • Send via SMTP                         │
   └────────┬────────────────────────────────┘
            │
            ▼
5. RETURN SUCCESS
   ┌─────────────────────────────────────────┐
   │ Return: "Check your email"              │
   └────────┬────────────────────────────────┘
            │
            ▼
6. USER RECEIVES EMAIL
   ┌─────────────────────────────────────────┐
   │ • Clicks link or copies code            │
   │ • Navigates to reset page               │
   └────────┬────────────────────────────────┘
            │
            ▼
7. SUBMIT RESET
   ┌─────────────────────────────────────────┐
   │ POST /auth/reset-password               │
   │ reset_code: "token"                     │
   │ new_password: "NewPass123"              │
   └────────┬────────────────────────────────┘
            │
            ▼
8. VALIDATE
   ┌─────────────────────────────────────────┐
   │ • Find user with matching code          │
   │ • Check code not expired                │
   │ • Validate new password (8+ chars)      │
   └────────┬────────────────────────────────┘
            │
            ├─── Code invalid/expired? ──► Return 400 Error
            │
            ▼
9. UPDATE PASSWORD
   ┌─────────────────────────────────────────┐
   │ • Hash new password                     │
   │ • Generate new salt                     │
   │ • Clear reset code                      │
   │ • Update in database                    │
   └────────┬────────────────────────────────┘
            │
            ▼
10. RETURN SUCCESS
    ┌─────────────────────────────────────────┐
    │ Return: "Password reset successfully"   │
    └────────┬────────────────────────────────┘
             │
             ▼
    11. FRONTEND
        ┌────────────────────────┐
        │ Show success message   │
        │ Redirect to login      │
        │ (after 2 seconds)      │
        └────────────────────────┘
```

## Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   FRONTEND LAYER                             │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │  LoginView     │  │ RegisterView   │  │ ForgotPwView │  │
│  │  .vue          │  │ .vue           │  │ .vue         │  │
│  └────────┬───────┘  └────────┬───────┘  └────────┬──────┘  │
│           │                   │                   │          │
│           └───────────────────┼───────────────────┘          │
│                               │                              │
│                               ▼                              │
│                    ┌─────────────────────┐                  │
│                    │   authStore.js      │                  │
│                    │   (Pinia Store)     │                  │
│                    │                     │                  │
│                    │ • user              │                  │
│                    │ • isAuthenticated   │                  │
│                    │ • setUser()         │                  │
│                    │ • logout()          │                  │
│                    │ • initializeAuth()  │                  │
│                    └──────────┬──────────┘                  │
│                               │                              │
│                      Router Guards Applied                   │
│                               │                              │
└───────────────────────────────┼──────────────────────────────┘
                                │
                    HTTP/HTTPS Requests
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                    API LAYER                                 │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            userRouter.py (Endpoints)                │   │
│  │                                                     │   │
│  │  • UserRegisterRouter()                            │   │
│  │  • UserLoginRouter()                               │   │
│  │  • UserMFAVerifyRouter()                           │   │
│  │  • UserLogoutRouter()                              │   │
│  │  • UserMeRouter()                                  │   │
│  │  • UserForgotPasswordRouter()                      │   │
│  │  • UserResetPasswordRouter()                       │   │
│  │  • UserEnableMFARouter()                           │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │                                         │
│  ┌────────────────▼────────────────────────────────────┐   │
│  │        userService.py (Business Logic)             │   │
│  │                                                     │   │
│  │  • hash_password()           ► PBKDF2-SHA256       │   │
│  │  • verify_password()         ► Compare hashes      │   │
│  │  • register_user()           ► Create + hash       │   │
│  │  • login()                   ► Auth + session      │   │
│  │  • generate_mfaCode()       ► Random 6 digits     │   │
│  │  • send_mfa_email()          ► SMTP               │   │
│  │  • request_password_reset()  ► Gen code + email   │   │
│  │  • reset_password()          ► Update password     │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │                                         │
│  ┌────────────────▼────────────────────────────────────┐   │
│  │        userRepo.py (Data Access)                   │   │
│  │                                                     │   │
│  │  • create_user()             ► INSERT              │   │
│  │  • get_user_by_email()       ► SELECT by email     │   │
│  │  • get_user_by_id()          ► SELECT by id        │   │
│  │  • update_lastLogin()       ► UPDATE timestamp    │   │
│  │  • set_mfaCode()            ► UPDATE MFA fields   │   │
│  │  • verify_mfaCode()         ► SELECT & compare    │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │                                         │
└───────────────────┼─────────────────────────────────────────┘
                    │
        Database & External Services
                    │
┌───────────────────▼─────────────────────────────────────────┐
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ MySQL        │  │ Filesystem   │  │ SMTP Service    │  │
│  │ Database     │  │ Sessions     │  │ Email Provider  │  │
│  │              │  │              │  │                 │  │
│  │ • Users      │  │ /flask_      │  │ • Gmail         │  │
│  │ • Timestamps │  │  sessions/   │  │ • Outlook       │  │
│  │ • Hashes     │  │              │  │ • Custom SMTP   │  │
│  │ • MFA data   │  │              │  │                 │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

## Database Table Relationships

```
Users Table
┌──────────────────────────────────────────────────┐
│ PK │ id                    │ INT auto_increment   │
├────┼───────────────────────┼──────────────────────┤
│    │ email                 │ VARCHAR(255) UNIQUE  │
│    │ firstName            │ VARCHAR(127)         │
│    │ lastName             │ VARCHAR(127)         │
│    │ passwordHash         │ VARCHAR(255)         │
│    │ salt                  │ VARCHAR(255)         │
│    │ lastLogin            │ DATETIME nullable    │
│    │ createdAt            │ DATETIME default now │
│    │ updatedAt            │ DATETIME on update   │
│    │ mfaEnabled           │ BOOLEAN default 0    │
│    │ mfaCode              │ VARCHAR(6) nullable  │
│    │ mfaCodeExpires      │ DATETIME nullable    │
└──────────────────────────────────────────────────┘

Session Storage (Filesystem)
/flask_sessions/
├── session_id_1.json
├── session_id_2.json
└── session_id_3.json

Example session content:
{
  "user_id": 1,
  "email": "user@example.com",
  "expires": 1707350400,
  "createdAt": 1707091200
}
```

## Security Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   PASSWORD SECURITY                          │
└──────────────────────────────────────────────────────────────┘

User Password Input
    │
    ▼
Client-side validation
    │ (8+ chars, not empty, etc)
    ▼
Server receives plaintext
    │
    ▼
Generate random salt
    │
    ▼
PBKDF2-HMAC-SHA256
    │ (100,000 iterations)
    ├─ password
    ├─ salt
    └─ returns: hash
    │
    ▼
Compare with stored hash
    │
    ├─ Match? ──► Login success
    │
    └─ No match? ──► Login failure

Never stored: plaintext password
Always stored: hash + salt (separate)

┌──────────────────────────────────────────────────────────────┐
│                   SESSION SECURITY                           │
└──────────────────────────────────────────────────────────────┘

Session Created
    │
    ▼
Store user_id in server-side session
    │
    ▼
Set session cookie
    │ • HttpOnly: ✓ (prevents JS access)
    │ • Secure: ✓ (HTTPS only in prod)
    │ • SameSite: Lax (CSRF protection)
    │ • Expires: 7 days (configurable)
    ▼
Browser stores cookie
    │
    ▼
Each request includes cookie
    │
    ▼
Server validates session
    │ ✓ Session exists
    │ ✓ Not expired
    │ ✓ User still exists
    ▼
Grant access to resource

Session cleanup: automatic on expiration

┌──────────────────────────────────────────────────────────────┐
│                   MFA SECURITY                               │
└──────────────────────────────────────────────────────────────┘

Login successful
    │
    ▼
Generate random 6-digit code
    │
    ▼
Set expiration (10 minutes)
    │
    ▼
Store in database (encrypted in prod)
    │
    ▼
Send email with code
    │
    ▼
Return user_id (not full user object)
    │
    ▼
User enters code within 10 minutes
    │
    ▼
Server verifies:
    │ ✓ Code matches stored code
    │ ✓ Code not expired
    │ ✓ User exists
    ▼
Clear code from database
    │
    ▼
Create full session
    │
    ▼
Grant access
```

---

**This diagram shows the complete architecture and security flow of the authentication system.**

