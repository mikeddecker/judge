# Jurysporten en AI skillherkenning: gedaan subjectiviteit?

Skillherkenning door Artificiële Intelligentie staat voor de deur. In jurysporten zoals rope skipping, artistieke gymnastiek of synchroonzwemmen zou er binnenkort wel eens een machinaal gestuurde juryassistent de routine kunnen beoordelen en zo helpen bij het bepalen van de score van de oefening.

Mike De Decker, een student toegepaste informatica aan de Hogeschool in Gent (HOGENT), creëerde voor zijn bachelorproef een juryassistent die rope skipping skills herkent in Double Dutch 3 (DD3) freestyles. DD3 is een specifiek onderdeel in rope skipping, waarbij er twee draaiers zijn en één springer. Je kan het vergelijken met de balk of ongelijke leggers in de gymnastiek of het hordelopen, verspringen of discuswerpen in de atletiek.

(foto DD3 example)

## Wat is rope skipping?

Rope skipping is net als artistieke gymnastiek of dans een sport met veel creativiteit en verschillende elementen. Om te bepalen wie de beste, mooiste en moeilijkste routine of freestyle heeft, is er een jurypanel die volgens de juryhandleiding de freestyle live of vertraagd bekijken om een numerieke score toe te kennen. Aangezien dit soms voor subjectiviteit of menselijke fouten kan zorgen, kan een computergestuurde assistent hierbij helpen, idealiter onder het toeziend oog van een menselijke jury-expert.

## Hoe werkt de assistent?

Naar voorbeelden van beeldherkenning, gebarentaalherkenning of NextJump's speedcounter onderzocht Mike in zijn bachelorproef de mogelijkheid tot het herkennen van meerdere verschillende skills in een volledige video. Hierbij waren er drie grote obstakels, die de rode draad in de gehele architectuur van het AI-model vormden. Rope skipping freestyles duren namelijk 60 tot 75 seconden en zijn een aaneenschakeling van verschillende skills, die variabel zijn in lengte en op verschillende afstand worden gefilmd. Hieronder worden deze drie grote stappen kort toegelicht. 

## Stap 1: Lokaliseren van de skippers

De eerste stap is het lokaliseren van de atleten. Deze zijn vaak gefilmd met statische camera's, of op verschillende zoomlevels, waardoor ze de ene keer dicht en de andere keer verder weg van de camera springen. Hierdoor kunnen ze uitgeknipt worden en kan elke pixel optimaal benut worden in de volgende stappen. Denk maar aan her verschil tussen het bekijken van een video op een mobiele telefoon ten opzichte van een computer op gelijke afstand. Het is toch makkelijker te zien wat er zich afspeelt op je computer?

(Afbeelding localize - before & after)

## Stap 2: Segmenteren van korte skill fragmenten.

Eenmaal ingezoomd kunnen we bepalen wanneer een skill start en eindigt. Dit concept is relatief eenvoudig, want doorgaans begint en eindigt een skill wanneer de springer de grond verlaat en terug op de grond landt. Immers moet er een touw onder kunnen. Deze momenten aanduiden en laten leren door de computer, zorgt voor freestyles te splitsen zijn in meerdere korte fragmenten die een skill zouden bevatten.

(Afbeelding fragmentjes/grafiek)

## Stap 3: Herkennen van de skill

Eenmaal we deze verschillende fragmenten hebben, kunnen we het AI model voorbeeld skills laten analyseren om ongeziene skills laten voorspellen. Echter waren hier twee moeilijkheden.

Ten eerste bestaan skills uit verschillende elementen. Zo heb je 3 atleten, waarvan 2 draaiers en 1 springer en geef je aan wat de springer doet, wat draaier één doet, wat draaier twee doet, hoeveel touwrotaties er waren... Het model berekent dan alle verschillende elementen/eigenschappen, op basis van wat het leerde uit voorbeeldskills.

Ten tweede gaat over de skills die variëren in lengte. De ene skill wordt sneller uitgevoerd dan de andere, waar het model niet mee overweg kan. We willen dus skill fragementen van gelijke lengte. In essentie is een video een opeenvolging van afbeeldingen, denk maar aan een stopmotion. Stel dat een handstand uit 18 opeenvolgende afbeeldingen bestaat en een salto uit 15, dan kan je ervoor kiezen om afbeeldingen te dupliceren of te knippen om gelijke fragmenten te bekomen. Deze uniforme vorm van data is bruikbaar door AI modellen om te berekenen welke skills in de freestyles van springers zitten.

(Afbeelding/timelaps skill)

## Post AI skill herkenning

Eenmaal je weet welke skills in de freestyle zitten, kan je ze omzetten naar een numerieke score volgens de vooropgestelde juryhandleiding. Onder toeziend oog van een jury expert, kunnen freestyles transparanter, objectiever, accurater en door minder juryleden beoordeeld worden.

Deze technologie maakt het tevens mogelijk om een live score te geven tijdens de freestyle of om het aantrekkelijker te maken voor het publiek, door skillfragmenten opnieuw af te spelen met naam erbij.

> Hoewel er nog veel foutieve voorspellingen zijn, door een gelimiteerde gelabelde dataset, begint de AI al basiselementen zoals een plankhouding (push-up), handstand (frog), split of salto te herkennen.

(Video, predicted skills)

## Toekomst van de assistent

Als je de bachelorproef van Mike zou lezen, merk je dat er nog veel optimalisatiemogelijkheden zijn. Zo bestaat de mogelijk om de lokalisatiestap met de segmentatiestap te combineren en zo veel meer.

Een interessante optimalisatie momenteel in testfase is het idee om het model toepasbaar te maken op meerdere events en sportoverschrijdend in één gehele dataset. Echter zijn er nog onvoldoende voorbeeldfragmenten, laat staan voorbeeldenfragmenten uit van andere sporten. De grafiek hieronder toont de stijgende accuraatheid van de herkenningsfase in de set-up van de bachelorproef die enkel focuste op één event, namelijk Double Dutch single freestyle (DD3). Daarbij heeft niet elke skill voldoende voorbeelden om uit te leren, aldus is hogere accuraatheid uitgesloten.
Zodra de juryassistent nauwkeurig genoeg is, zal het consitent jureren over verschillende freestyles heen. Immers ondervind een AI model geen invloed van vermoeidheid, meningsverschillen tussen juryleden over correcte uitvoering, invloed de naam/club/team tijdens het jureren, invloed van bekende namen, regels die je vergeet of verkeerd onthoud, onoplettendheid enzovoort. Dit verhoogt de betrouwbaarheid van het jureren.

(Afbeelding: Grafiek van de accuracy)

Deze technologie is niet enkel interessant voor rope skipping, maar ook voor gymnastiek, synchroonzwemmen, kunstschaatsen en andere sporten. Verder is het een flexibele technologie die aangepast kan worden naar de vereiste toepassing.

