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

# 🔒 Security (by design?)

![[AI-Judge.excerpt.Security-summary]]

More details in [[AI-Judge.Security-by-design]]
# 🕶 Privacy by design

![[AI-Judge.Privacy-by-design-summary]]

# ✴ Availability

![[AI-Judge.Availability]]

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
No real plan, for now just think about:
- Backup
- How to set-up/run the app again after server breakdown.
	- code README.md
	- ...

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
    - I assume you have an api now, maybe merging projects now is the way to go?
- WEB app, Idk if you have one right now?
    - Can remain mostly as-is -> expand on it (dev ideas...)
    - 'Quick' setup: old pc -> only web + api
        - Then develop the database/file sync? (I have 2 external SSD's of 1TB of video's now)
        - When my pc is online -> sync video's/labels/database

# 📑 Action points
- [ ] ⏫ Walk through general page
- [ ] ⏫ Discuss the discussion points
- [ ] ⏫ Send Eva email about recording on competitions
    - Yes, she wrote something about Gymfed(leden) -> gymfed videos

I will propose -> training can still be on the same dataset.

