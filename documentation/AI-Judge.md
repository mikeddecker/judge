# Goal


# Content
```table-of-contents
```
# Project stage: development

What does this mean? Most aspects are still in the development stage. Even though a lot is built, much more features are waiting in line to be developed.

# High level overview

- Browse videos
- Label frames (localization)
- Segment videos
- Recognize (skill) elements in te segment 

# 🔒Security (by design?)

![[AI-Judge.excerpt.Security-summary]]



More details in [[AI-Judge.Security-by-design]]

- Privacy by design (if we want to comply with EU GDPR):
    - Videos we can use (permission vs privacy notice - borderline to be decided/defined)
        - Uploaded through the nextjump app or site? -> This builds us a big data.
        - Videos provided by IJRU, AMJRU, GymFed?
        - For GymFed it is still under the agreement of it being a school project -> but that has finished.
    - Opt-out option
    - [https://gdpr.eu/checklist/](https://gdpr.eu/checklist/)
    - AI-Act -> limited risk AI -> nearing towards or is even a high risk AI (especially when it replaces judges and humans mainly watch)
- Availability
    - usa & belgium?
    - down/incidents = down? -> best effort principle?
    - sync between servers & videos
- Back-up
    - By hosting both in the usa & be?
- Monitoring
- Integration into nextjump (subdomain on nextjump.app?)
    - Do I register [nextjump.be](http://nextjump.be) as well? (about 10 euro/year)
- Computervision
    - Training possible on two sides
    - I think you can focus on this part a bit more?
    - I think there is a bug in the numeric label train/test labels
    - Maybe review it together?
    - Reads out the database -> provided labels.
    - Do we directly integrate/migrate this towards NextJump?
    - Ideas I had was instead of pre-filtering based on full json string -> filtering on layer value occurrence. If then a label is a null tensor (or a fully masked one > skip)
    - Next idea, calculate acc/f1 based on tags/output_heads instead of full layers -> which gives more info about accuracy of crosses depending on the rotation.
    - Review how to label wraps/DD-transitions/snappers/skillsegments having multiple rope rotations, but which are not multiples.
    - While we're at it, review the current setup of the layercomposition. It might be able to be slightly more easier.
- API -> for querying/posting/deleting videos, folders, users, labels.
    - I assume you have an api now, maybe merging projects now is the way to go?
- WEB app, Idk if you have one right now?
    - Can remain mostly as-is -> expand on it (dev ideas...)
    - 'Quick' setup: old pc -> only web + api
        - Then develop the database/file sync? (I have 2 external SSD's of 1TB of video's now)
        - When my pc is online -> sync video's/labels/database