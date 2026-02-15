## Code improvement ideas
#TODO add changes prior to 6 July
- cv - JobExecutor - Fix connection to DB to fetch new Jobs (fresh connection)

+/- up until eind december
- [ ] Update single rope QUADS/TRiPLES EB AB, sASo -> sEB, EB, o, AS & s, o, AS
	- [ ] It becomes the rotation that is finishing the skill level
		- [ ] e.g. QUAD sEB, EB, o, CL
		- [ ] e.g. QUAD sEB, EB, EBs, CL
		- [ ] e.g. QUAD sEB, EBs, CL, CL
		- [ ] e.g. QUAD sEB, EBs, CL, o
		- [ ] e.g. QUAD s, EB, ...?
	- [ ] Current Skill ID since rule = 1370
		- [ ] 0
## Suggestions

![[AI-Judge.excerpt.Security-features-backlog]]

## Docker
- [ ] ⏫  Fix docker setup of prod -> web does not launch properly


# Domain models
- [ ] Use uuid to store

# Folders
- [ ] Store user uploaded video's into user -> videos
- [ ] Show/Define tag order to browse
# Videos
- [x] Explore drive to list videos ✅ 2025-02-01
- [ ] Tags: accept proposed tags
- [ ] RBAC: Allow users to view videos
# Tags
- [ ] 🔽 Auto add tags to videos during drive explore
- [ ] 🔽 Add/display tag to video 🆔 tagToVid
- [ ] 🔽 Propose tags based
- [ ] ⏬ Learn/propose tags based on video
## Stats/Results
- [ ] 🔽 Stats/results based on tags, users 🆔 k8lipo ⛔ tagToVid
- [ ] 🔽 Stats/results based on output_heads ⛔ tagToVid
- [ ] General stats page
	- [ ] Videos
	- [ ] Videos / tag (group)
	- [ ] Video total duration
	- [ ] Avg video duration distribution
	- [ ] Video total frames
	- [ ] Video fps / video division

# Jobs
- [ ] Consider marking jobs as “claimed” to avoid duplicate processing in future
- [ ] Train page
	- [ ] Show available models
		- [ ] Make Segmentation work
		- [ ] Make 
	- [ ] Show records having null properties (e.g. Feet: null)
	- [ ] Show process of current training step, batch
		- [ ] Localize
		- [ ] Segment
		- [ ] Skills

# Models
- [ ] Train using labels of specified user
- [ ] Train using labels of allowed users
- [ ] Localization
	- [ ] Add available localization methods
	- [ ] Train/Validate on different localization methods
	- [ ] Train/Validate on more models than YOLO only
	- [ ] Train using labels of specified user
	- [ ] Train using labels of allowed users
	- [ ] Suggest high confident predicted labels
	- [ ] ⏬ Edit label: foreground <-> background
	- [ ] ⏬ Edit label: adjust borders
- [ ] Segmentation
	- [ ] Learn recognition borders: start, end and whether it's a skill or not
	- [ ] Predict segments
	- [ ] Possibility to use localization
		- [ ] Select localization method (include mix)
		- [ ] Select model (best)
	- [ ] Train using labels of specified user
	- [ ] Train using labels of allowed users
	- [ ] Suggest high confident predicted labels
	- [ ] Distinct colors for segments, num skill instances, partial skill labels, full skill labels
	- [x] Segment without adding recognition info ✅ 2025-12-15
	- [ ] Move segment
		- [ ] Left border
		- [ ] Right border
		- [ ] Merge
		- [ ] Split

- [ ] SegLo
	- [ ] Model to combine these two steps?
	- [ ] Train using labels of specified user
	- [ ] Train using labels of allowed users
	- [ ] Suggest high confident predicted labels
- [ ] Recognition
	- [ ] Predict skills
	- [ ] Train using labels of specified user
	- [ ] Train using labels of allowed users
	- [ ] ⏫ Check these TODO's 
		- [ ] Segmented skill = named property (order by inverse relative occurence)
		- [ ] Show the current labeled videos, which are not not fully segmented with a skill property
		- [ ] Show the current labeled videos, which are fully segmented, order by skill property density

# Judging
- [ ] Add rulesets
- [ ] Add score/level based on composition
	- [ ] Based on current segment
	- [ ] Based on (tag+) previous (two?) segments
	- [ ] Based on tag
- [ ] Combine rulesets to a total score
- [ ] Add score based on assigned score/level
- [ ] Add recognition short name
- [ ] Define repetitions (based on previous segment?)
- Recognition -> level
- Recognition -> score
- Filter invalid skills = e.g. false multiples, unwell executed powers (labeled them with a PointWorthy value -> so filter)
- Define which skill/recognition is equal (repetitions)
- Checkbox to count repetitions or not (freestyle vs speed)
- Combine score aspects (difficulty, variation... -> seperation of concerns)
- ...

# Label suggestions
- [ ] 🔽 Localize suggestions: accept/deny
- [ ] 🔽 Segment suggestions: accept/deny
- [ ] 🔼 Recognition: Suggest high confident predicted labels (profile setting)
- [ ] 🔼 Recognition: Suggest/highlight predictions where it predicted something which occurs less often
- [ ] On the page where labels are now created
	- [ ] Add option to 'recognise while segment'
	- [ ] Segment the skills
	- [ ] Meanwhile they get recognised
	- [ ] Then label skills (+ add some sort of highlight/pre-filled value)
		- [ ] e.g. when the predicted value not the most occurring value e.g. a power variant ratio is much more equal compared to fault ratio)
		- [ ] e.g. when the value is mostly filled in / not masked.
        - (this could correspond to some sort of similarity embedding -> using/predicting skill masks)
        - (gymnastics are mostly filled in when one happens, sometimes indicated as None, to let it learn None is possible, other times it is masked in learning.)
        - Idem with fault, above someone (DD4), cross values for the 5th or 6th rotation...

# Code quality
- [ ] Debugging
	- [ ] CI/CD (Docker test)
		- [ ] Create other docker services testing the application with a limited dataset
			- [x] Launch web ✅ 2026-02-08
			- [x] Launch API ✅ 2026-02-08
			- [ ] Insert dummy values using API
			- [ ] Insert dummy jobs
			- [ ] Execute dummy jobs
			- [ ] Add property null values, should be filtered
			- [ ] ...
	- [ ] Create a run log showing example run values
		- [ ] Example label to tensor
		- [ ] Example max_composition_amounts
		- [ ] Example masking
		- [ ] Example skill length
		- [ ] ...
- [ ] datetime.now(timezone.utc)
- [ ] Type recipe

# Documentation
- [ ] Add pictures of a push-up, frog, split...
- [ ] Add a graph display speed records
- [ ] WEB: Add graphs displaying distribution of skills, rotations, type, hands, feet...
- [ ] Skillverdeling


## Snippets
```python
from datetime import datetime, timezone
datetime.now(timezone.utc)
```
