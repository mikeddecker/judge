# Voorstel in markdown

## Introduction

Jump rope evolving sport.
Year after year, an increasing amount of high level competitors are pushing the limits of jump rope.
% TODO : source?
Resulting in new skills, new combinations, better phyiques, beter rope material, faster movements.
In order for the judges to keep up with the jumpers and to correctly asses scores to a routine, Double Dutch freestyles are replayed at half speed on International competitions or even locally in Belgium.

Head judges around the world question the best way to correctly judge athletes as to give an accurate ranking on national or international competitions.
Many solutions have been provided: other judging rule-set, splitting judge responsibility, replaying the routine at half speed...
However, with the increasing popularity of image recognition, more powerful computers, applications that recognize objects in images \autocite{Singh_Gill_2022} or applications detecting simple human actions \autocite{LUQMAN_2022}, they wondered about the possibility to incorporate Artificial Intelligence into jump rope to assist judging routinees.

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

To put into perspective, safely guessing DD3 data = 286-357 video's of length 1 to 1min15 (2022-2024 Belgium teams, virtual world contest, two other world livestreams)
Individual freestyles from 2024 own recordings only will amount to a minimum of 782 freestyles (includes 2 camera perspectives, so about half of it different routines). So collecting more individual freestyles will amount to a minimum of 1000 freestyles or minimum 15h of freestyle data, whereas DD3 freestyles are close to 5h of freestyles.

#### Which discipline will be focused on?

Only one discipline for this thesis will be chosen.
There are two main disciplines to chose from. Either single freestyles (Single Rope, SR) or Double Dutch Single Freestyles (DD3)
Given the previous calculation and availability of data, it seems individual freestyles seem easier to get and the most likely path, however, in first thought, the variety and possible skills also increases, resulting in a more complex labeling (see skillmatrix later). One possibility for both DD3 & SR is to omit certain special cases simplifying the entire problem in a minimum viable product (MVP).
Skill label example DD3 & Single freestyle

Pro's & con's

|   Area   |    SR    |   DD3   |
| :------- | :------- | :------ |
| #Freestyles | 782-1000+| 286-352 |
| Hours    | 13h-16h+ | 5-6h    |
| MVP      | basic crosses, releases, multiples, gyms... | basic powers, gyms, turnerskills |
| skillmatrix | more complex | simpeler compared to SR |
| longer sequences | / | / |
| individuals      | 1 | 3 |
| competitions     | Okt-Nov | March-Apr |

Both datasets are skewed, there are common skills which can be found in 30-60% of the routines, where others occur only once, twice each competition, if they ocur at all.

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
