# Voorstel in markdown

## Introduction

Jump rope evolving sport.
Year after year, an increasing amount of high level competitors are pushing the limits of jump rope.
% TODO : source?
Resulting in new skills, new combinations, better phyiques, beter rope material, faster movements.
In order for the judges to keep up with the jumpers and to correctly asses scores to a routine, Double Dutch freestyles are replayed at half speed on International competitions or even locally in Belgium.

Head judges around the world question the best way to correctly judge athletes as to give an accurate ranking on national or international competitions.
Many solutions have been provided: other judging rule-set, splitting judge responsibility, replaying the routine at half speed...
However, with the increasing popularity of image recognition, more powerful computers, applications that recognize objects in images \autocite{Singh_Gill_2022} or applications detecting simple human actions \autocite{LUQMAN_2022}, they wondered about the possibility to incorporate Artificial Intelligence into jump rope to assist judging routines.

### Different disciplines

Speed, SR, SR2, DD3, DD4, CW
Short mention of levels and scores. (source IJRU/Gymfed?)
Each freestyle has a length of 1 to 1min15

### What are the challenges for judges today?

Increased skillset, combinations, speed, ...
Miscounts with mistakes (speed)
DD3 half speed.

### Which techniques can we apply to increase the accuracy of judging?

Further describe different rule-set, replay...
Aso describe SR judging, as to do a better literature related to both individual and team freestyles.

### How can we integrate machine learning into jump rope?

The preferred solution in this propasal is an AI-model recognizing skills in a freestyle.
In august 2023, NextJump tested their AI-speedcounter on the world competition.
Next step freestyles.
These skills can then be mapped to their corresponding level and summed up to achieve the score of the freestyle.

#### Which data is available for the AI to use?

Both individual and team freestyles are mostly recorded on competitions. All A-level team freestyles from the last three years (2022, 2023 & 2024) are already available and provided by Gymfed (Belgium) for training. Additionally some clubs posted some of their competition videos on social media which provides more data from the last 8 years and other camera perspectives.

To give more perspective, safely guessing DD3 data = 286-357 video's of length 1 to 1min15 (2022-2024 Belgium teams, virtual world contest, two other world livestreams)
Individual freestyles from 2024 own recordings only will amount to a minimum of 782 freestyles (includes 2 camera perspectives, so about half of it different routines). So collecting more individual freestyles will amount to a minimum of 1000 freestyles or minimum 15h of freestyle data, whereas DD3 freestyles are close to 5h of freestyles.

#### Which discipline will be focused on?

Only one discipline for this thesis will be chosen.
There are two main disciplines to choice from. Either single freestyles (Single Rope, SR) or Double Dutch Single Freestyles (DD3)
Given the previous calculation and availability of data, it seems individual freestyles seem easier to get and the most likely path, however, in first thought, the variety and possible skills also increases, resulting in a more complex labeling (see skillmatrix later). One possibility for both DD3 & SR is to omit certain special cases simplifying the entire problem in a minimum viable product (MVP).
Skill label example DD3 & Single freestyle

Pro's & con's

|   Area   |    SR    |   DD3   |
| :------- | :------- | :------ |
| #Freestyles | 782-1000+| 286-352 |
| Hours    | 13h-16h+ | 5-6h    |
| Years    | 782 in 2024 | mainly 2020-2024 |
| MVP      | basic crosses, releases, multiples, gyms... | basic powers, gyms, turnerskills |
| Levelguessing | 0 to 8 | 0 to 6-8 |
| Theoretical level limit | 8+ levels possible | 8+ levels possible |
| Variation elements | 6 | 4 |
| skillmatrix | more complex | simpeler compared to SR |
| longer sequences | / | / |
| individuals      | 1 | 3 |
| competitions     | Okt-Nov | March-Apr |

Both datasets are skewed, there are common skills which can be found in 30-70% of the routines (educated broad guess based on experience), where others occur only once, twice each competition, if they'll ocur at all.

#### What would be a minimal Proof of Concept (PoC)?

The MVP would be a model recognizing the most common skills.
This means omitting or just marking special combinations, longer double dutch switches or longer time sequences of emptiness in general.
By starting with a simplified notation, hereafter called pointified notation, dotted notation or skillmatrix/table representation, all (sub)skills can be mapped to this general skillrecognition notation/table.
Afterwards, levels can be given to each (sub)skill in the matrix.
Based on my own experiences, a quick guess would be, at least, 90-95% or more skills on competitions could be labeled using the simplified notation, not including special cases.

### Unanswered questions

Until now, only the scope has been narrowed. The literature, implementation & labeling will give more clarification on different questions.

- How will the model be built?
- What would be the main structure of the model?
- Which human activity recognition examples can be used or altered as the base of the model?
- When are AI-recognitions acceptable to potentially use on competitions?
- How much data is expected to increase the accuracy off the Judge.
- How can we use the AI-Judge to improve judges?
- What needs to changed to a working model, to apply it on other judgesports such as gymnastics, synchronized swimming...

## Literature

Small overview

- Jump rope intro: Explaining some disciplines, SR, DD, CW, and mention other variants. (SR2, SR4, DD3, DD4, speed, DU, TU, box...).
- Skills intro: define (sub)skills we want to label
- Computervision, techniques & available models.
- Skillmatrix

### Jump rope intro

Disciplines...

### Skills intro

Translate 2.1, 2.1.1, 2.1.2

### Computer vision

Computer vision because availability of videographic material.
The studyfield related to images, videos... is computer vision.
More specific HAR: **(Pareek & Thakkar, 2020)**

Human Gait Recognition **(Alharthi et al., 2019)** & Human pose estimation **(Song et al., 2021)**. Pose recognition can be used to assist a full activity recognition.

### HAR general progress

TODO find paper, reference to come to steps or use pareek again?

Step 3 is an in between to make the progress trackable and can be omitted.

    \item Jumper Localisation
    \item Skill segmentation, start of skill, end of skill
    \item Predict level or variation element (multiple, cross, gym...)
    \item Predict the effective skill

### Jumper localization (2.4)

CNN good for working with image data,
Based on paper X, paper Y, ... would be good.
Like **Zaidi et al., 2021** (survey) talks about different models, YOLO, SSD, CenterNet and their variants. Based on the convolutions.

Explain VGG-16

Could be assisted by seperating foreground and background. This is called VOS or VIS. **Gao et al. (2022)** compares and discusses some models.

Transferlearning on a pre-trained-model like ResNet or GoogleNet?

Smoothing out box predictions

TODO : shortlist like densepose, VGG-16, ResNet...

Or models like [cutie](https://arxiv.org/pdf/2310.12982), [github-cutie](https://github.com/hkchengrex/Cutie), densepose or meta sam2 can be used. Densepose can even provide bouding boxes of the main poses deteced. Perhaps just a convex hull and some padding will be enough to smart crop the images.

### Video action segmentation

Split video in small identifiable actions, like skills or subskills.
Zahan et al. (2023) translate 2.8

add: smoothing info

See [paperswithcode](https://paperswithcode.com/task/action-segmentation),
[LTContext](https://arxiv.org/pdf/2308.11358v2) (cooking)

TODO : find clear implementation or relatively followable instruction for usage. (Or don't read when tired)

Additional pre-processing like masking the jumpers, using denspepose or cutie can be used.

### Skillrecognition (2.5)

After segmented action videos.
Temporal & space information. To recognize skills themselves.
Walk the path & verify / find (better) sources
CNN-LSTM
LSTM-CNN
convLSTM
SAM (Self-Attention convLSTM) (2020)
SAM vergeleek met MIM (2019) - Memory in Memory
read/translate 2.6

footnote: the same or other models, like densepose with clear bodypart distinction could/will be used/tried to assist recognizing the skills.

TODO : verify if convLSTM/SAM/MIM are FCN's
TODO : add more understanding of transformers (using self attention?)

### Skill complexity & levels

Now that we better now a part of the information, let's talk more about skills.
Translate 2.7
split in general info
2.7.1 single rope & single rope skillmatrix (translate)
2.7.2 double dutch singles & skillmatrix (TODO)

maybe 2.7.1 - 2.7.4

### Group activities (wrong order)

Translation, maybe omit Issa & Shanable (2023) and add tip of the veil volleybal (spikes, servings, waiting, talking...)

TODO : research stagNet
[group-activity-volleybal](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mengshi_Qi_stagNet_An_Attentive_ECCV_2018_paper.pdf)

### Unknown activities (wrong order)

Skillmatrix should cover it, but ... zero-shot

TODO : Research zero-shot learning (unknown activities)
[zeroshot](https://www.researchgate.net/publication/339028828_Zero-Shot_Human_Activity_Recognition_Using_Non-Visual_Sensors)

## Methodology

Modify & translate to follow user story map

Proposal now or version 2 in :

1) Determine shortlist Localization: like Denspose, VGG-16 based model or YOLO or ... (max 3) + argument choses why?
2) Additional model for video action segmentation besides LTContext
3) Try search more recent models for recognition other than SAM (using transformers?), maybe skip MiM or normal convLSTM (not putting in shortlist). SAM seems to be the current first choice
4) Additional or deeper search in SAM2 for second alternative like densepose. (more difference in mask of person, arms, legs...) to assist localization, segmentation & recognition. (Applies to all 3 steps.)
5) Better understand stagNet (group activities) as DD3 is a team recognition. (Improving model.)

Thus, following user story map. ([canva-story-map](https://www.canva.com/design/DAGVz44QCgc/_Mr9BrOqwwdy9cf-ieYFVg/edit?utm_content=DAGVz44QCgc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton))

Mind that releases are not bound.

Release 1 : general + label location

1) overview videos: navigate, filter, rename
2) view video info, could have edit
3) label inappropriate/blurry moments (idea: could be used, rather not)
4) label passage/empty (livestream/wait) moments (idea: no skills)
5) label jumper location
6) Statistics of the general data distribution

Release 2 : predict location

1) visualize jumpoer location predictions
2) edit borders from new predictions
3) visualize biggest localization mistakes (within video)
4) visualize biggest localization mistakes (over all videos)
5) localization model statistics
6) data augmentation (if needed)

Release 3 : label action segments + label skills

1) Label video action segment
2) Loop-replay action segment
3) Label segmented skills names
4) mark false skills
5) skills to level mapper (should have)
6) search skills by name (could have)
7) mark execution scale (could have)
8) skilldistribution statistics (+/- distributionmatrix)

Release 4.1 : predict action segments

1) visualize & compare AI segmented skills
2) visualize AI segmented skills from new videos (with confidence levels?)

Release 4.2 : predict skills

1) model predicts levels (1-8 classes)
2) model predicts variation element (4/6 classes)
3) model predicts skill
4) visualize & compare AI labeled skills & compare with total (judge)score
5) validate AI recognized skills & save as new training input
6) label AI recognized skills as new or "i dont know"
7) action segmentation stats
8) skillrecognition stats

Release Bis

1) Language
2) add competition jump order (easy recordings)
3) stats competition
4) insert multiple judge systems

### add talk about accuracy/metric judge vs ai

Comparing judge scores on competion vs (correctly) labeled freestyle. (Should have some difference), calculate average difference = target.

### Training & Hardware

Laptop has GPU.
Ideal: Ask server at school (with GPU) (TODO), training can be done at night (low peak usage)

## Expected results

- Localization - perfectly doable
- Video action segmention, could probably pose a problem as results could not be what's to be expected. But if it the segmentation is slowly increasing nearing the end of the paper, I'll be happy.
- Recognizing the skill actually seems slightly less or equal in difficulty compared to segmenting the actions.
- When skills are starting to be recognized, expected results are to be close to the scores of the judges or real label.

## Expected conclusion

translate
