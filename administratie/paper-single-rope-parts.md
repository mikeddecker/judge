# Omitted parts paper

To either replace from intro to literature or to only side note at the end (Single rope)

## Intro

### What are the challenges of jump rope today

As described earlier, jumpers are improving quickly throughout the years. To elaborate this idea, it is mentioned in one of the IJRU 2023 livestreams that quadruple unders where 10 to 15 years ago rather unique, last years quintuple unders are becoming the norm. This is also true for Double Dutch freestyles or speed scores. With the current judging rule-set for double dutch, routines are replayed at half speed to increase judging accuracy on competitions.

### What can be applied to increase judging accuracy

The current panel consists of multiple judges (10-12), evaluating 5 different aspects of a routine at the same time. Mostly two or three people are judging the difficulty of the routine. As it was hard to keep up, the half-speed-replay was added to correctly assign a score.
Using the replay system, more time is needed to evaluate a routine, requiring additional difficulty certified judges on competitions. To put this further in perspective, a competition has 2 judging panels to improve competition flow and wait times. When the replay is applied, panel A could judge single rope, panel B double dutch where there are two diff panels within the second panel to alternate between watching live, replaying and counting the total score, while the other difficulty panel could be replaying, counting and the watching live. This results in around 1 judge for each 10 participating jumpers. Even more for team competitions.

Other than the replay system, different judging-rules can be applied, the current rule system in Belgium dates from 2023, requiring all judges to do a new exam to be certified again.

So splitting responsibility, fine-tuning the rule-set or replaying are already in use. The question arose with the rise of Artificial Intelligence, if machine learning could be applied to judge freestyles. We know AIs are not perfect, so are judges. Both can make mistakes while assigning levels or deductions to a routine.

### How can we integrate AI in freestyles

As of august 2023, \href{https://nextjump.app/}{NextJump} tested their AI-speed-counter on the world competition acquiring, really accurate results\footnote{95\% off by one}. Using this as a base and examples of human action recognition \autocite{Yin_2024}, the likelihood to succeed implementing skill-recognition in freestyles grows.
When skills are recognized, they can be mapped to their corresponding level and summed up to achieve the total score of a freestyle.

### Which data

All A-level team routines from the last three years (2022, 2023 \& 2024) are already available and provided by Gymfed (Belgium) for training. Additionally some clubs post their competition videos on social media which provides more data from the last 8 years and other camera perspectives.
To give more perspective, safely guessing DD3 data = 286-357 video's of length 1 to 1min15 (2022-2024 Belgium teams, virtual world contest \& two other world livestreams)
However, individual freestyles are more abundant. Calculating only my own recordings from 2024 will amount to a minimum of 782 freestyles (includes 2 camera perspectives, so about half of it are different routines). Additionally, collecting more individual freestyles will quickly amounts to a minimum of 1000 routines or a 16h+ of freestyle data, whereas DD3 freestyles are close to 5h. The NextJump speed counter used 10h of speed data to achieve their current results.

### Which discipline will be focused on?

Only one discipline for this thesis will be chosen.
There are two main disciplines to choice from. Either single freestyles (Single Rope, SR) or Double Dutch Single Freestyles (DD3)
Given the previous calculation and availability of data, it seems individual freestyles seem easier to get and the most likely path, however, in first thought, the variety and possible skills also increases, resulting in a more complex labeling (see skillmatrix later). One possibility for both DD3 \& SR is to omit certain special cases simplifying the entire problem in a minimum viable product (MVP). My educated guess based on experience would be more omitting for SR compared to DD3. See tabel \ref{tbl:data-comparison}

\begin{table*}[t]
    \centering
    \begin{tabular}{|l|l|l|}
        \hline
        & SR & DD3 \\ \hline
        \#Freestyles & 782-1000+ & 286-352 \\ \hline
        Hours & 13h-16h+ & 5-6h \\ \hline
        Years & 782 in 2024 & mainly 2020-2024 \\ \hline
        MVP & basic variation elements & basic powers, gyms, turnerskills \\ \hline
        Level-guessing & 0 to 8 & 0 to 8 \\ \hline
        Theoretical level limit & 8+ levels possible & 8+ levels possible \\ \hline
        Variation elements & 6 & 4 \\ \hline
        skillmatrix & more complex & simpeler compared to SR \\ \hline
        longer sequences & / & / \\ \hline
        individuals & 1 & 3 \\ \hline
        competitions & Oct-Nov & March-Apr \\ \hline
    \end{tabular}
    \caption{Data comparison}
    \label{tbl:data-comparison}
\end{table*}

Both datasets are skewed, there are common skills which can be found in 30-70\% of the routines (educated broad guess based on experience), where others occur only once, twice each competition, if they'll occur at all.

### How is diff scored

This can be an isolated skill, the base, a transition or usage in a other skills like multiples. A simple example would be the quadruple under EB AS (level 6), where you get three levels from the quadruple under, one from the EB and two from the AS. Each performed level has its own score, e.g. nine points for a level six skill\footnote{For A category in Belgium age 16+}. Diff judges look at a skill and calculate/memorize the total level of the skill and write it down when seen in a routine.

### MVP / POC

By starting with a simplified notation, hereafter called pointified notation, dotted notation or skillmatrix/table representation, all (sub)skills can be mapped to this general skillrecognition notation/table.
Afterwards, levels can be given to each (sub)skill in the matrix.
An educated, quick guess based on my own experiences would be that, at least, 85-95\% or more skills on competitions could be labeled using the simplified notation, not including special cases.

